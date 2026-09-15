//! "Hershey font" synthetic data: treat our own real single-character ink
//! recordings as a library of reusable glyph strokes (the way a Hershey
//! font is just a stroke path per glyph) and recombine them into brand-new
//! words that never appeared in the real corpus — unlike `bin/augment.rs`
//! (which perturbs the *shape* of words we already have), this generates
//! new *vocabulary*: new letter sequences and transitions for CTC to learn
//! from.
//!
//! Deliberately built from our own real pen strokes, not a traced TTF
//! outline (the approach a reference Python script used): real strokes
//! carry real velocity/pressure-adjacent timing dynamics a font-traced
//! contour can't, and letters recombined this way still look like this
//! person's handwriting, not a robotic font trace.
//!
//! Glyph coverage is whatever the corpus happens to contain as
//! single-character samples — as of this corpus, that's all 26 lowercase
//! letters (unevenly: some have 40-80 real instances, most rare ones have
//! only 1-4) and almost nothing else (a couple of uppercase letters, one
//! digit). So synthesized words are lowercase-only; this does not (and
//! can't) manufacture glyphs for characters we've never actually seen
//! written. `bin/augment.rs` remains the tool for digit/uppercase/
//! punctuation diversity, since it perturbs whatever real samples with
//! those characters already exist rather than requiring a glyph per class.
//!
//! Usage:
//!   synth_glyphs --out PATH --count N [--wordlist PATH] [SOURCE...]

use std::collections::HashMap;
use std::path::PathBuf;

use rand::seq::IndexedRandom;
use rand::{Rng, RngExt, SeedableRng};

use hwr_ink::ink::Ink;
use hwr_model::augment::{augment, AugmentConfig};
use hwr_model::corpus::{load_source, save_pairs};

/// Isolate single-character `(text, ink)` samples from a loaded corpus and
/// group them by character — our glyph library. Multiple real instances of
/// the same letter are kept (not averaged/deduplicated): picking a random
/// one per use is itself a free source of natural variation, on top of
/// `augment`'s synthetic perturbation.
fn build_glyph_library(pairs: &[(String, Ink)]) -> HashMap<char, Vec<Ink>> {
    let mut lib: HashMap<char, Vec<Ink>> = HashMap::new();
    for (text, ink) in pairs {
        let mut chars = text.chars();
        if let (Some(c), None) = (chars.next(), chars.next()) {
            if !ink.is_empty() {
                lib.entry(c).or_default().push(ink.clone());
            }
        }
    }
    lib
}

/// Uniformly rescale `ink` (around its own centroid, both axes together —
/// aspect-preserving, matching `spline.rs::prepare`'s own normalization
/// philosophy) so its height matches `target_height`. Real single-character
/// samples were captured at whatever pen/zoom scale was in effect at the
/// time, which varies sample to sample; without this, glued-together
/// letters would have wildly inconsistent relative sizes.
fn rescale_to_height(ink: &Ink, target_height: f32) -> Ink {
    let height = ink.y_range.max - ink.y_range.min;
    if height < 1e-3 {
        return ink.clone();
    }
    let factor = target_height / height;
    let cx = (ink.x_range.min + ink.x_range.max) * 0.5;
    let cy = (ink.y_range.min + ink.y_range.max) * 0.5;
    let mut out = Ink::new();
    for stroke in ink.strokes() {
        for p in stroke {
            out.push((p.x - cx) * factor + cx, (p.y - cy) * factor + cy, p.z);
        }
        out.pen_up();
    }
    out
}

/// Glue single-character glyph inks into one word-shaped `Ink`, left to
/// right, with a jittered horizontal advance and a shared (per-word)
/// baseline offset per glyph — analogous to `make_word` in the reference
/// Python script, but every glyph is a real stroke recording instead of a
/// font-contour trace, and inter-glyph timing comes from `Ink::append`
/// (which already handles the time-offset/stroke-bookkeeping bridge
/// between two `Ink`s correctly, so we don't have to re-derive it here).
fn make_word(
    word: &str,
    library: &HashMap<char, Vec<Ink>>,
    augment_config: &AugmentConfig,
    rng: &mut impl Rng,
) -> Option<Ink> {
    const TARGET_HEIGHT: f32 = 1.0;

    let mut result = Ink::new();
    let mut cursor_x = 0.0f32;
    let n = word.chars().count();
    for (i, c) in word.chars().enumerate() {
        let templates = library.get(&c)?;
        let template = templates.choose(rng)?;

        let mut glyph = rescale_to_height(template, TARGET_HEIGHT);
        glyph = augment(&glyph, augment_config, rng);

        let width = (glyph.x_range.max - glyph.x_range.min).max(0.05);
        // Slight, occasionally-negative gap: real handwriting sometimes
        // lets adjacent letters' strokes overlap a touch (e.g. "ll", "oo"),
        // rather than always leaving clean whitespace.
        let gap = rng.random_range(-0.06..0.22) * TARGET_HEIGHT;
        // Small per-glyph baseline jitter, plus a mild word-level drift so
        // the whole word isn't perfectly level (real writing rarely is).
        let baseline_jitter = rng.random_range(-0.05..0.05) * TARGET_HEIGHT;
        let drift = 0.04 * TARGET_HEIGHT * (i as f32 / n.max(1) as f32 - 0.5)
            * rng.random_range(-1.0..1.0);

        let dx = cursor_x - glyph.x_range.min;
        // Recenter the glyph's own vertical midpoint at 0, then apply the
        // word-level baseline jitter/drift on top.
        let dy = baseline_jitter + drift - (glyph.y_range.min + glyph.y_range.max) * 0.5;
        glyph = glyph.translate(cgmath::Vector2::new(dx, dy));

        let time_offset = if i == 0 {
            0.0
        } else {
            rng.random_range(0.02..0.15)
        };
        result.append(glyph, time_offset);

        cursor_x += width + gap;
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// A small built-in fallback word list (common short English words), used
/// only if no on-disk dictionary is found — keeps the tool usable on a
/// machine without `/usr/share/dict/words` (Linux CI, etc.), just with
/// less vocabulary variety.
const FALLBACK_WORDS: &str = include_str!("../fallback_words.txt");

fn load_wordlist(path: Option<&PathBuf>) -> Vec<String> {
    let candidates = [
        PathBuf::from("/usr/share/dict/words"),
        PathBuf::from("/usr/dict/words"),
    ];
    let text = if let Some(p) = path {
        std::fs::read_to_string(p).unwrap_or_else(|e| panic!("failed to read {}: {e}", p.display()))
    } else if let Some(found) = candidates.iter().find(|p| p.exists()) {
        std::fs::read_to_string(found).unwrap_or_default()
    } else {
        FALLBACK_WORDS.to_string()
    };
    text.lines()
        .map(|w| w.trim().to_lowercase())
        .filter(|w| w.len() >= 2 && w.len() <= 12 && w.chars().all(|c| c.is_ascii_lowercase()))
        .collect()
}

/// A random lowercase string of `len` characters drawn from whatever
/// characters the glyph library actually covers — pure letter-transition
/// diversity, independent of real English word structure, to complement
/// the dictionary-word samples above.
fn random_string(len: usize, alphabet: &[char], rng: &mut impl Rng) -> String {
    (0..len).map(|_| *alphabet.choose(rng).unwrap()).collect()
}

fn main() {
    let mut out_path = PathBuf::from("synthetic_glyphs.txt");
    let mut count = 5000usize;
    let mut wordlist_path: Option<PathBuf> = None;
    let mut seed = 9002u64;
    let mut sources: Vec<PathBuf> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out_path = PathBuf::from(args.next().expect("--out needs a path")),
            "--count" => {
                count = args
                    .next()
                    .expect("--count needs a number")
                    .parse()
                    .expect("count must be a number")
            }
            "--wordlist" => wordlist_path = Some(PathBuf::from(args.next().expect("--wordlist needs a path"))),
            "--seed" => {
                seed = args
                    .next()
                    .expect("--seed needs a number")
                    .parse()
                    .expect("seed must be a number")
            }
            other => sources.push(PathBuf::from(other)),
        }
    }
    if sources.is_empty() {
        sources.push(PathBuf::from("armrest/data/inks"));
    }

    let mut pairs = Vec::new();
    for source in &sources {
        load_source(source, &mut pairs);
    }
    let library = build_glyph_library(&pairs);
    let mut alphabet: Vec<char> = library.keys().copied().collect();
    alphabet.sort();
    println!(
        "Glyph library: {} characters covered ({:?}), {} total real single-char samples",
        alphabet.len(),
        alphabet.iter().collect::<String>(),
        library.values().map(|v| v.len()).sum::<usize>()
    );
    if alphabet.is_empty() {
        eprintln!("No single-character samples found in the given source(s) — nothing to build a glyph library from.");
        std::process::exit(1);
    }

    let mut words = load_wordlist(wordlist_path.as_ref());
    words.retain(|w| w.chars().all(|c| library.contains_key(&c)));
    println!(
        "Wordlist: {} usable words (fully covered by the glyph library)",
        words.len()
    );

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Lighter than augment.rs's default: these glyphs are already being
    // recombined into a novel arrangement, so we don't want to also
    // distort each one as aggressively as when perturbing an already-whole
    // real word.
    let glyph_augment_config = AugmentConfig {
        max_rotation: 0.08,
        scale_range: (0.9, 1.1),
        max_shear: 0.12,
        jitter_amplitude: 0.02,
        jitter_frequency: 1.0,
        time_warp: 0.1,
    };

    let mut out = Vec::with_capacity(count);
    let mut attempts = 0usize;
    while out.len() < count && attempts < count * 4 {
        attempts += 1;
        // 70% real dictionary words (realistic English letter structure),
        // 30% random strings over the covered alphabet (raw transition
        // diversity a curated wordlist under-represents).
        let text = if !words.is_empty() && rng.random_bool(0.7) {
            words.choose(&mut rng).unwrap().clone()
        } else {
            let len = rng.random_range(2..=9);
            random_string(len, &alphabet, &mut rng)
        };
        if let Some(ink) = make_word(&text, &library, &glyph_augment_config, &mut rng) {
            out.push((text, ink));
        }
        if out.len() % 2000 == 0 && out.len() > 0 {
            println!("generated {}/{count}", out.len());
        }
    }

    println!(
        "Generated {} synthetic glyph-recombination samples, writing to {}",
        out.len(),
        out_path.display()
    );
    save_pairs(&out_path, &out).expect("failed to write synthetic corpus");
}
