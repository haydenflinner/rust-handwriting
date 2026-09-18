//! Shared locations for app-generated calibration data.
//!
//! Native writes `calibration.txt` under the OS data dir. The wasm build
//! keeps the same `text\tink` lines in `localStorage` instead.

use hwr_ink::ink::Ink;
use hwr_model::corpus;

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
/// Where calibration mode appends accepted samples, and where review mode
/// reads them back from.
pub fn calibration_file_path() -> PathBuf {
    let dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hwr");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("calibration.txt")
}

#[cfg(target_arch = "wasm32")]
const CALIBRATION_KEY: &str = "hwr.calibration";

pub fn append_calibration_sample(text: &str, ink: &Ink) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = calibration_file_path();
        let result = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut file| writeln!(file, "{text}\t{ink}"));
        if let Err(err) = result {
            eprintln!(
                "failed to save calibration sample to {}: {err}",
                path.display()
            );
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut body = load_calibration_text();
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(text);
        body.push('\t');
        body.push_str(&ink.to_string());
        body.push('\n');
        if let Err(err) = set_calibration_text(&body) {
            crate::log(&format!("failed to save calibration sample: {err}"));
        }
    }
}

pub fn load_calibration_pairs() -> Vec<(String, Ink)> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        match corpus::load_pairs(calibration_file_path()) {
            Ok(pairs) => pairs,
            Err(_) => Vec::new(),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        corpus::parse_pairs(&load_calibration_text())
    }
}

#[cfg(target_arch = "wasm32")]
fn load_calibration_text() -> String {
    let Some(window) = web_sys::window() else {
        return String::new();
    };
    let Ok(Some(storage)) = window.local_storage() else {
        return String::new();
    };
    storage
        .get_item(CALIBRATION_KEY)
        .ok()
        .flatten()
        .unwrap_or_default()
}

#[cfg(target_arch = "wasm32")]
fn set_calibration_text(text: &str) -> Result<(), String> {
    let window = web_sys::window().ok_or_else(|| "no window".to_string())?;
    let storage = window
        .local_storage()
        .map_err(|_| "localStorage unavailable".to_string())?
        .ok_or_else(|| "localStorage unavailable".to_string())?;
    storage
        .set_item(CALIBRATION_KEY, text)
        .map_err(|_| "failed to write localStorage".to_string())
}
