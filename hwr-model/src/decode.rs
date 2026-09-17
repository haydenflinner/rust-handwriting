//! CTC decoding: turn a `[steps, classes]` softmax output into text.
//!
//! Ported from armrest's `ml.rs` — this part is plain math over `&[f32]` and
//! has no dependency on any particular inference framework.

use std::cmp::Ordering;
use std::collections::HashMap;

/// The alphabet the model was trained on: a leading space, digits, letters,
/// then ASCII punctuation. Index `CHARS.len()` (one past the end) is the CTC
/// blank class, so `CLASSES = CHARS.len() + 1`.
pub const CHARS: &str = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

pub fn classes() -> usize {
    CHARS.len() + 1
}

/// Map `text` to class indices into `CHARS`, or `None` if it contains a
/// character outside the trained alphabet.
pub fn encode_labels(text: &str) -> Option<Vec<usize>> {
    let chars: Vec<char> = CHARS.chars().collect();
    text.chars()
        .map(|c| chars.iter().position(|&x| x == c))
        .collect()
}

pub trait ModelOutput {
    type Out;
    fn read_from(&self, buffer: &[f32]) -> Self::Out;
}

pub struct RawOutput;

impl ModelOutput for RawOutput {
    type Out = Vec<f32>;

    fn read_from(&self, buffer: &[f32]) -> Self::Out {
        buffer.to_vec()
    }
}

pub struct Greedy;

impl ModelOutput for Greedy {
    type Out = String;

    fn read_from(&self, buffer: &[f32]) -> String {
        greedy_decode(buffer)
    }
}

pub struct Beam<L> {
    pub size: usize,
    pub language_model: L,
}

impl<L: LanguageModel> ModelOutput for Beam<L> {
    type Out = Vec<(String, f32)>;

    fn read_from(&self, buffer: &[f32]) -> Vec<(String, f32)> {
        let chars: Vec<_> = CHARS.chars().collect();
        beam_decode(buffer, self.size, &chars, &self.language_model)
    }
}

pub fn greedy_decode(buffer: &[f32]) -> String {
    let index_to_char: Vec<_> = CHARS.chars().collect();
    let char_count = index_to_char.len() + 1;
    let mut res = String::new();
    let mut last_char = index_to_char.len();
    for i in 0..(buffer.len() / char_count) {
        let offset = i * char_count;
        let max: usize = (0..char_count)
            .max_by(|j, k| {
                buffer[offset + j]
                    .partial_cmp(&buffer[offset + k])
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap();
        if max < index_to_char.len() && max != last_char {
            res.push(index_to_char[max]);
        }
        last_char = max
    }
    res
}

pub trait LanguageModel {
    fn odds(&self, prefix: &str, ch: char) -> f32;
    fn odds_end(&self, _prefix: &str) -> f32 {
        1.0
    }
}

impl LanguageModel for &[char] {
    fn odds(&self, _prefix: &str, ch: char) -> f32 {
        if self.contains(&ch) {
            1.0
        } else {
            0.0
        }
    }
}

impl LanguageModel for bool {
    fn odds(&self, _prefix: &str, _: char) -> f32 {
        if *self {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct P {
    blank: f32,
    nonblank: f32,
}

impl P {
    fn one() -> P {
        P {
            blank: 1.0,
            nonblank: 0.0,
        }
    }

    fn zero() -> P {
        P {
            blank: 0.0,
            nonblank: 0.0,
        }
    }

    fn total(self) -> f32 {
        self.blank + self.nonblank
    }
}

pub fn beam_decode(
    buffer: &[f32],
    beam_width: usize,
    alphabet: &[char],
    lm: &impl LanguageModel,
) -> Vec<(String, f32)> {
    use partial_sort::PartialSort;

    let blank = alphabet.len();
    let classes = blank + 1;

    let mut beams = vec![(vec![], P::one())];

    let mut candidates = HashMap::<Vec<usize>, P>::new();

    for step in buffer.chunks_exact(classes) {
        for (char, p_char) in step.iter().enumerate() {
            for (prefix, p_curr) in &beams {
                // TODO: quite a lot of copying in here! Maybe fine for short sequences?
                let prefix_string: String = prefix.iter().map(|c| alphabet[*c]).collect();
                if char == blank {
                    let p_next = candidates.entry(prefix.to_vec()).or_insert(P::zero());
                    p_next.blank += p_curr.total() * p_char;
                } else {
                    let mut prefix_plus_char = prefix.clone();
                    prefix_plus_char.push(char);
                    if prefix.last() == Some(&char) {
                        // This is the repeat case!
                        // Calculate odds both when it's a real repeat (ie. has a blank in between)
                        // as well as the merging case.
                        let p_merged = candidates.entry(prefix.to_vec()).or_insert(P::zero());
                        // FIXME: I'm not confident that I'm applying the language model correctly here.
                        // should the RHS here be multiplied by lm_odds as well? (If not why not?)
                        p_merged.nonblank += p_curr.nonblank * p_char;

                        let p_repeat = candidates.entry(prefix_plus_char).or_insert(P::zero());
                        let lm_odds = lm.odds(&prefix_string, alphabet[char]);
                        p_repeat.nonblank += p_curr.blank * p_char * lm_odds;
                    } else {
                        // It's a different char... we care about total probability only.
                        let p_next = candidates.entry(prefix_plus_char).or_insert(P::zero());
                        let lm_odds = lm.odds(&prefix_string, alphabet[char]);
                        p_next.nonblank += p_curr.total() * p_char * lm_odds;
                    }
                }
            }
        }

        beams.clear();
        beams.extend(candidates.drain());
        let to_sort = beam_width.min(beams.len());
        beams.partial_sort(to_sort, |(_, left), (_, right)| {
            right.total().partial_cmp(&left.total()).expect("NaN???")
        });
        beams.truncate(beam_width);
    }

    let mut result: Vec<_> = beams
        .iter()
        .map(|(beam, p)| {
            let string = beam.iter().map(|&c| alphabet[c]).collect::<String>();
            let odds = lm.odds_end(&string);
            (string, p.total() * odds)
        })
        .collect();

    result.sort_by(|(_, p0), (_, p1)| p1.partial_cmp(p0).expect("NAN???"));

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_singleton() {
        let buffer = [0f32, 1f32, 0f32];

        let chars = ['a', 'b'];

        let result = beam_decode(&buffer, 20, &chars, &true);

        assert_eq!(&result[0].0, "b")
    }

    #[test]
    fn test_beam_merges() {
        // NB: three ways to get "a", so it wins even though blank is always more likely.
        let buffer = [
            0.2f32, 0.0f32, 0.8f32, // ...
            0.4f32, 0.0f32, 0.6f32,
        ];

        let chars = ['a', 'b'];

        let result = beam_decode(&buffer, 20, &chars, &true);

        assert_eq!(&result[0].0, "a")
    }
}
