use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{OnceLock, RwLock};

// rustuna only interns keys whose cardinality is typically low, such as parameter names and
// AttrKey strings. We do not currently have a lifecycle where interned keys become unreachable
// and need to be reclaimed, so a small process-global interner is sufficient here without
// depending on a more feature-rich crate.
static INTERNER: OnceLock<RwLock<StringInterner>> = OnceLock::new();

fn interner() -> &'static RwLock<StringInterner> {
    INTERNER.get_or_init(|| RwLock::new(StringInterner::default()))
}

#[derive(Default)]
struct StringInterner {
    map: HashMap<&'static str, u32>,
    strings: Vec<&'static str>,
}

#[derive(Eq, Clone, Copy, Debug, PartialEq)]
/// Interned string used for low-cardinality keys such as parameter names and attribute names.
pub struct InternedString(u32);

impl InternedString {
    /// Returns the interned string as a `'static` string slice.
    pub fn as_str(&self) -> &'static str {
        interner().read().unwrap().resolve(self.0)
    }

    fn new(value: &str) -> Self {
        if let Some(symbol) = interner().read().unwrap().get(value) {
            return InternedString(symbol);
        }

        let mut interner = interner().write().unwrap();
        if let Some(symbol) = interner.get(value) {
            return InternedString(symbol);
        }
        InternedString(interner.intern(value))
    }
}

impl StringInterner {
    fn get(&self, value: &str) -> Option<u32> {
        self.map.get(value).copied()
    }

    fn intern(&mut self, value: &str) -> u32 {
        let symbol = u32::try_from(self.strings.len()).expect("too many interned strings");
        let leaked = Box::leak(value.to_owned().into_boxed_str());
        self.map.insert(leaked, symbol);
        self.strings.push(leaked);
        symbol
    }

    fn resolve(&self, symbol: u32) -> &'static str {
        self.strings[symbol as usize]
    }
}

impl fmt::Display for InternedString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl AsRef<str> for InternedString {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for InternedString {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl Hash for InternedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl PartialOrd for InternedString {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InternedString {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl From<&str> for InternedString {
    fn from(value: &str) -> Self {
        InternedString::new(value)
    }
}

impl From<String> for InternedString {
    fn from(value: String) -> Self {
        InternedString::new(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interned_string_reuses_symbol() {
        let a = InternedString::from("owner");
        let b = InternedString::from("owner".to_string());

        assert_eq!(a, b);
        assert!(std::ptr::eq(a.as_str(), b.as_str()));
    }
}
