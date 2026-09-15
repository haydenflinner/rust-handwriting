//! Shared on-disk locations for app-generated data.

use std::path::PathBuf;

/// Where calibration mode appends accepted samples, and where review mode
/// reads them back from.
pub fn calibration_file_path() -> PathBuf {
    let dir = dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("hwr");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("calibration.txt")
}
