//! Timestamps for trials.
//!
//! [`crate::trial::PersistedTrial`] carries timezone-naive **UTC** datetimes. UTC needs a clock but
//! no timezone database, so this module is built on `std::time` alone and every storage backend can
//! stamp a trial without taking on a dependency.
//!
//! Converting to the local time that users see is deliberately left to the outermost layer: the
//! Python bindings do it with Python's own `datetime` module, so Rustuna and Optuna always agree on
//! the timezone database, and Rust callers can pick whichever crate they already use.

/// Seconds in a day.
const SECS_PER_DAY: i64 = 86_400;

/// A point in time broken down into UTC calendar fields.
struct UtcParts {
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    microsecond: u32,
}

/// Reads the wall clock as whole seconds since the Unix epoch plus microseconds within that second.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn unix_now() -> (i64, u32) {
    use std::time::{SystemTime, UNIX_EPOCH};

    // A clock reading before the Unix epoch would mean the machine is misconfigured by more than
    // fifty years; clamping to the epoch avoids threading an error through every storage write for
    // a case that cannot happen in practice.
    let since_epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (since_epoch.as_secs() as i64, since_epoch.subsec_micros())
}

/// Reads the wall clock as whole seconds since the Unix epoch plus microseconds within that second.
///
/// `std::time::SystemTime::now` panics on `wasm32-unknown-unknown`, which has no clock of its own.
/// The host always has one, so it is read through `Date.now()`; the resolution is milliseconds, so
/// the microseconds are always a multiple of 1000.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn unix_now() -> (i64, u32) {
    unix_from_millis(js_sys::Date::now())
}

/// Splits milliseconds since the Unix epoch into whole seconds and microseconds within the second.
///
/// Compiled outside wasm as well so that the conversion above is covered by the ordinary test run.
#[cfg(any(all(target_arch = "wasm32", target_os = "unknown"), test))]
fn unix_from_millis(millis_since_epoch: f64) -> (i64, u32) {
    // `floor` rather than a cast, so that a clock set before 1970 keeps the microseconds positive
    // and the seconds land on the earlier whole second.
    let secs = (millis_since_epoch / 1000.0).floor();
    let micros = ((millis_since_epoch - secs * 1000.0) * 1000.0) as u32;
    (secs as i64, micros.min(999_999))
}

impl UtcParts {
    fn now() -> UtcParts {
        let (secs, microsecond) = unix_now();
        UtcParts::from_unix(secs, microsecond)
    }

    fn from_unix(secs: i64, microsecond: u32) -> UtcParts {
        // Euclidean division so that instants before the epoch floor towards the earlier day
        // instead of truncating towards zero.
        let days = secs.div_euclid(SECS_PER_DAY);
        let secs_of_day = secs.rem_euclid(SECS_PER_DAY);
        let (year, month, day) = civil_from_days(days);
        UtcParts {
            year,
            month,
            day,
            hour: (secs_of_day / 3600) as u32,
            minute: (secs_of_day % 3600 / 60) as u32,
            second: (secs_of_day % 60) as u32,
            microsecond,
        }
    }
}

impl std::fmt::Display for UtcParts {
    /// Formats as `%Y-%m-%d %H:%M:%S.ffffff`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let UtcParts {
            year,
            month,
            day,
            hour,
            minute,
            second,
            microsecond,
        } = *self;
        write!(
            f,
            "{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}.{microsecond:06}"
        )
    }
}

/// Converts a count of days since 1970-01-01 into a UTC calendar date.
///
/// This is Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Shift the era so that the cycle starts on 0000-03-01, which puts the leap day at the end of
    // the year and makes the month arithmetic below branch-free.
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = (shifted - era * 146_097) as u64; // [0, 146096]
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365; // [0, 399]
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100); // [0, 365]
    let month_prime = (5 * day_of_year + 2) / 153; // [0, 11], where 0 is March
    let day = (day_of_year - (153 * month_prime + 2) / 5 + 1) as u32; // [1, 31]
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    } as u32; // [1, 12]
    let year = year_of_era as i64 + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// Returns the current time as a timezone-naive UTC timestamp.
///
/// The format is `%Y-%m-%d %H:%M:%S.ffffff`, which matches `str(datetime)` in Python and is what
/// [`crate::trial::PersistedTrial`] holds. Microseconds are always written in full so that
/// `datetime.fromisoformat` accepts the value on every supported Python version, and so that the
/// fixed width leaves the timestamps orderable as plain strings.
pub fn now_naive_utc() -> String {
    UtcParts::now().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn days_in_month(year: i64, month: u32) -> u32 {
        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 => 29,
            2 => 28,
            _ => unreachable!("month out of range: {month}"),
        }
    }

    fn format_unix(secs: i64, micros: u32) -> String {
        UtcParts::from_unix(secs, micros).to_string()
    }

    #[test]
    fn formats_the_unix_epoch() {
        assert_eq!(format_unix(0, 0), "1970-01-01 00:00:00.000000");
    }

    #[test]
    fn formats_known_instants() {
        // Cross-checked against `date -u -r <secs>`.
        assert_eq!(format_unix(1, 2), "1970-01-01 00:00:01.000002");
        assert_eq!(format_unix(86_399, 999_999), "1970-01-01 23:59:59.999999");
        assert_eq!(format_unix(86_400, 0), "1970-01-02 00:00:00.000000");
        assert_eq!(format_unix(951_782_400, 0), "2000-02-29 00:00:00.000000");
        assert_eq!(format_unix(1_709_164_800, 0), "2024-02-29 00:00:00.000000");
        assert_eq!(format_unix(4_107_542_400, 0), "2100-03-01 00:00:00.000000");
        assert_eq!(format_unix(1_735_689_599, 0), "2024-12-31 23:59:59.000000");
        assert_eq!(format_unix(1_735_689_600, 0), "2025-01-01 00:00:00.000000");
        // Before the epoch, where truncating division would land on the wrong day.
        assert_eq!(format_unix(-1, 0), "1969-12-31 23:59:59.000000");
        assert_eq!(format_unix(-86_400, 0), "1969-12-31 00:00:00.000000");
    }

    #[test]
    fn matches_a_calendar_walked_independently_over_four_centuries() {
        // 1970-01-01 through 2369, covering every leap-year rule including the 400-year exception.
        // Walking the calendar day by day is an independent implementation of the same arithmetic:
        // it shares nothing with civil_from_days beyond the leap-year rule itself.
        let mut days = 0i64;
        for year in 1970..2370i64 {
            for month in 1..=12u32 {
                for day in 1..=days_in_month(year, month) {
                    assert_eq!(civil_from_days(days), (year, month, day), "day {days}");
                    days += 1;
                }
            }
        }
        assert_eq!(days, 146_097, "four centuries are 146097 days");
    }

    #[test]
    fn matches_a_calendar_walked_backwards_from_the_epoch() {
        let mut days = 0i64;
        for year in (1600..1970i64).rev() {
            for month in (1..=12u32).rev() {
                for day in (1..=days_in_month(year, month)).rev() {
                    days -= 1;
                    assert_eq!(civil_from_days(days), (year, month, day), "day {days}");
                }
            }
        }
    }

    #[test]
    fn milliseconds_from_a_host_clock_are_split_correctly() {
        // The wasm path reads `Date.now()`, which yields milliseconds as an f64.
        assert_eq!(unix_from_millis(0.0), (0, 0));
        assert_eq!(unix_from_millis(1.0), (0, 1_000));
        assert_eq!(unix_from_millis(999.0), (0, 999_000));
        assert_eq!(unix_from_millis(1_000.0), (1, 0));
        assert_eq!(
            unix_from_millis(1_709_164_800_123.0),
            (1_709_164_800, 123_000)
        );
        // Before the epoch, where a plain cast would truncate towards zero and lose a second.
        assert_eq!(unix_from_millis(-1.0), (-1, 999_000));
        assert_eq!(unix_from_millis(-1_000.0), (-1, 0));

        let (secs, micros) = unix_from_millis(1_709_164_800_123.0);
        assert_eq!(format_unix(secs, micros), "2024-02-29 00:00:00.123000");
        let (secs, micros) = unix_from_millis(-1.0);
        assert_eq!(format_unix(secs, micros), "1969-12-31 23:59:59.999000");
    }

    #[test]
    fn now_is_a_full_width_naive_utc_timestamp() {
        let naive = now_naive_utc();
        assert_eq!(naive.len(), "1970-01-01 00:00:00.000000".len(), "{naive}");
        assert_eq!(naive.as_bytes()[10], b' ', "{naive}");
        // A timestamp taken later never sorts earlier, which is what the fixed width buys.
        assert!(naive <= now_naive_utc());
    }
}
