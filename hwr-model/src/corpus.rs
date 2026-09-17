//! Loading armrest's `text\tink` corpus format (see `armrest/README.md`),
//! e.g. `armrest/data/inks/*.txt`. Shared by the recognition smoke-test and,
//! later, by pretraining/calibration.

use std::io::{self, BufRead};
use std::path::Path;

use hwr_ink::ink::Ink;

/// Read `(text, ink)` pairs from a file in armrest's tab-separated ink format.
pub fn load_pairs(path: impl AsRef<Path>) -> io::Result<Vec<(String, Ink)>> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut pairs = Vec::new();
    for line in reader.lines() {
        pairs.extend(parse_line(&line?));
    }
    Ok(pairs)
}

/// Parse `(text, ink)` pairs from an in-memory corpus, e.g. one embedded
/// with `include_str!` — same format as [`load_pairs`], no filesystem
/// access, so it also works in a compiled app that doesn't ship the raw
/// corpus file.
pub fn parse_pairs(text: &str) -> Vec<(String, Ink)> {
    text.lines().filter_map(parse_line).collect()
}

fn parse_line(line: &str) -> Option<(String, Ink)> {
    if line.is_empty() {
        return None;
    }
    let (text, ink_str) = line.split_once('\t')?;
    Some((text.to_string(), Ink::from_string(ink_str)))
}

/// Load `(text, ink)` pairs from a source, which may be a single `.txt`
/// file (in the format read by [`load_pairs`]) or a directory of them —
/// used to accept either shape uniformly on the command line (e.g. `train`,
/// `augment`).
pub fn load_source(path: &Path, pairs: &mut Vec<(String, Ink)>) {
    if path.is_dir() {
        let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(path)
            .unwrap_or_else(|e| panic!("failed to read corpus dir {}: {e}", path.display()))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "txt"))
            .collect();
        entries.sort();
        for entry in entries {
            load_source(&entry, pairs);
        }
    } else {
        match load_pairs(path) {
            Ok(file_pairs) => {
                println!("Loaded {} lines from {}", file_pairs.len(), path.display());
                pairs.extend(file_pairs);
            }
            Err(e) => eprintln!("Skipping {}: {e}", path.display()),
        }
    }
}

/// Load each source, apply [`cap_long_samples`], and optionally truncate
/// *per source* after a seeded shuffle. Use `per_source_cap` when mixing a
/// huge corpus (MNIST digits) with a smaller one (glyph words) so the
/// concat-then-truncate path cannot drown the small source. The final
/// concatenation is shuffled with seed 1234, matching `bin/train.rs`.
pub fn load_sources_capped(
    sources: &[impl AsRef<Path>],
    max_steps: usize,
    per_source_cap: Option<usize>,
) -> Vec<(String, Ink)> {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut pairs = Vec::new();
    for (i, source) in sources.iter().enumerate() {
        let mut one = Vec::new();
        load_source(source.as_ref(), &mut one);
        one = cap_long_samples(one, max_steps);
        if let Some(cap) = per_source_cap {
            let mut rng = StdRng::seed_from_u64(1234 + i as u64);
            one.shuffle(&mut rng);
            one.truncate(cap);
            one.shrink_to_fit();
        }
        pairs.extend(one);
    }
    let mut rng = StdRng::seed_from_u64(1234);
    pairs.shuffle(&mut rng);
    pairs
}

/// Write `(text, ink)` pairs in the same `text\tink` format `load_pairs`
/// reads, one per line.
pub fn save_pairs(path: impl AsRef<Path>, pairs: &[(String, Ink)]) -> io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    for (text, ink) in pairs {
        writeln!(file, "{text}\t{ink}")?;
    }
    Ok(())
}

/// Split a multi-word `(text, ink)` sample into one sample per word, using
/// the largest horizontal gaps between strokes as word boundaries (there
/// are exactly `words.len() - 1` breaks needed among `strokes.len() - 1`
/// candidate gaps, so this just takes the biggest ones — punctuation glued
/// to a word, e.g. "turtle,", stays glued to it since the gap right before
/// it is small). Falls back to returning the pair unchanged if there's only
/// one word, or too few strokes to split one-per-word.
///
/// Exists to cap sequence length: a handful of samples in armrest's corpus
/// are full sentences or entire poem lines (up to ~450 encoded steps, vs.
/// ~20-30 for a typical word), and those long outliers are what made
/// per-batch training time explode under Burn's autodiff (see
/// `hwr-model`'s `train` module docs) — splitting them into their
/// constituent words directly removes the long tail, and arguably better
/// matches how the app is actually used (one word/phrase per box).
pub fn split_into_words(text: &str, ink: &Ink) -> Vec<(String, Ink)> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= 1 {
        return vec![(text.to_string(), ink.clone())];
    }

    let strokes: Vec<_> = ink.strokes().collect();
    if strokes.len() < words.len() {
        return vec![(text.to_string(), ink.clone())];
    }

    let mut gaps: Vec<(usize, f32)> = (0..strokes.len() - 1)
        .map(|i| {
            let prev_end_x = strokes[i].last().unwrap().x;
            let next_start_x = strokes[i + 1][0].x;
            (i, next_start_x - prev_end_x)
        })
        .collect();
    gaps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut boundaries: Vec<usize> = gaps
        .into_iter()
        .take(words.len() - 1)
        .map(|(i, _)| i)
        .collect();
    boundaries.sort_unstable();

    let mut groups = Vec::with_capacity(words.len());
    let mut start = 0;
    for &b in &boundaries {
        groups.push(start..=b);
        start = b + 1;
    }
    groups.push(start..=(strokes.len() - 1));

    words
        .into_iter()
        .zip(groups)
        .map(|(word, range)| {
            let mut out = Ink::new();
            for stroke in &strokes[range] {
                for p in *stroke {
                    out.push(p.x, p.y, p.z);
                }
                out.pen_up();
            }
            (word.to_string(), out)
        })
        .collect()
}

/// Apply [`split_into_words`] to every sample whose encoded step count
/// exceeds `max_steps`, leaving shorter samples (the vast majority)
/// untouched.
pub fn cap_long_samples(pairs: Vec<(String, Ink)>, max_steps: usize) -> Vec<(String, Ink)> {
    let mut out = Vec::with_capacity(pairs.len());
    let mut split_count = 0usize;
    for (text, ink) in pairs {
        let steps = crate::spline::encode_vec(&ink).len() / crate::spline::WIDTH;
        if steps > max_steps {
            split_count += 1;
            out.extend(split_into_words(&text, &ink));
        } else {
            out.push((text, ink));
        }
    }
    if split_count > 0 {
        println!(
            "Split {split_count} sample(s) over {max_steps} steps into per-word samples ({} total after split)",
            out.len()
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes a fake word as `letters` tiny strokes, each offset a little to
    /// the right of the last, with a `gap`-sized jump before the word
    /// starts (so `split_into_words` can find the boundary).
    fn write_word(ink: &mut Ink, start_x: f32, letters: usize) -> f32 {
        for i in 0..letters {
            let x = start_x + i as f32 * 0.5;
            ink.push(x, 0.0, x as f32);
            ink.push(x + 0.2, 1.0, x as f32 + 0.1);
            ink.pen_up();
        }
        start_x + (letters.max(1) - 1) as f32 * 0.5 + 0.2 // rightmost x used
    }

    #[test]
    fn split_into_words_matches_text_token_count() {
        let mut ink = Ink::new();
        let end1 = write_word(&mut ink, 0.0, 3); // "cat"
        let end2 = write_word(&mut ink, end1 + 3.0, 4); // "dogs" — big gap before it
        let _end3 = write_word(&mut ink, end2 + 3.0, 5); // "mouse"

        let words = split_into_words("cat dogs mouse", &ink);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].0, "cat");
        assert_eq!(words[1].0, "dogs");
        assert_eq!(words[2].0, "mouse");
        // Every stroke from the original ink should show up in exactly one word.
        let total_strokes: usize = words.iter().map(|(_, ink)| ink.strokes().count()).sum();
        assert_eq!(total_strokes, ink.strokes().count());
    }

    #[test]
    fn split_into_words_single_word_is_unchanged() {
        let mut ink = Ink::new();
        write_word(&mut ink, 0.0, 3);
        let words = split_into_words("cat", &ink);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].0, "cat");
    }
}
