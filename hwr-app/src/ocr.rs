//! The recognizer, loaded once at startup from the pretrained checkpoint
//! or (with `--features vlm`) from a local HunyuanOCR directory.

use std::path::PathBuf;

use bevy::prelude::*;
use hwr_ink::ink::Ink;
use hwr_model::Recognizer;

/// Pretrained on armrest's bundled `data/inks/*.txt` corpus — see
/// `hwr-model/src/bin/train.rs`. Embedded so a packaged app doesn't depend
/// on a runtime file path (matching how armrest embeds `english_ascii.tflite`).
/// Burn's Named-MessagePack format (see `train::save`) — not the old
/// candle-era `model.safetensors`, stale since the Burn migration.
///
/// Development runs prefer a file on disk (see [`checkpoint_candidates`])
/// so `hwr-app` can pick up the current PBT `--out` without a rebuild.
static CHECKPOINT: &[u8] = include_bytes!("../../checkpoints/model.mpk");

/// `NonSend` because candle's CPU tensors aren't required to be `Sync`, and
/// we only ever touch the recognizer from the main thread anyway.
pub struct OcrRecognizer(pub OcrEngine);

pub enum OcrEngine {
    Hat(Recognizer),
    #[cfg(feature = "vlm")]
    Vlm(crate::vlm::VlmOcr),
}

impl OcrRecognizer {
    pub fn recognize(&self, ink: &Ink) -> Result<String, String> {
        match &self.0 {
            OcrEngine::Hat(rec) => rec.recognize_greedy(ink).map_err(|err| err.to_string()),
            #[cfg(feature = "vlm")]
            OcrEngine::Vlm(vlm) => vlm.recognize(ink),
        }
    }

    pub fn is_vlm(&self) -> bool {
        match &self.0 {
            OcrEngine::Hat(_) => false,
            #[cfg(feature = "vlm")]
            OcrEngine::Vlm(_) => true,
        }
    }
}

/// Path or label of the weights that were loaded, shown in test mode so
/// it's obvious which experiment the app is scoring against.
#[derive(Resource)]
pub struct OcrCheckpointSource(pub String);

pub struct OcrPlugin;

impl Plugin for OcrPlugin {
    fn build(&self, app: &mut App) {
        // PreStartup so test-mode UI can read `OcrCheckpointSource` on Startup.
        app.add_systems(PreStartup, setup_recognizer);
    }
}

fn setup_recognizer(world: &mut World) {
    let (recognizer, source) = load_recognizer();
    eprintln!("ocr: loaded {source}");
    world.insert_resource(OcrCheckpointSource(source));
    world.insert_non_send(recognizer);
}

fn requested_backend() -> String {
    std::env::var("HWR_OCR_BACKEND")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
}

fn load_recognizer() -> (OcrRecognizer, String) {
    let backend = requested_backend();
    match backend.as_str() {
        "vlm" | "hunyuan" | "hunyuanocr" | "paddle" | "paddleocr-vl" | "paddleocr_vl" => {
            load_vlm_or_explain()
        }
        "hat" | "ctc" => load_hat(),
        "" => {
            // With the vlm feature, prefer the local VLM when its checkpoint
            // is on disk so `cargo run --features vlm` is enough to try it.
            #[cfg(feature = "vlm")]
            {
                if vlm_model_dir().is_some() {
                    return load_vlm_or_explain();
                }
                eprintln!(
                    "ocr: no HunyuanOCR dir found (set HWR_VL_MODEL_DIR or put weights in models/HunyuanOCR); using HAT"
                );
            }
            load_hat()
        }
        other => {
            eprintln!("ocr: unknown HWR_OCR_BACKEND={other:?}; using HAT");
            load_hat()
        }
    }
}

fn load_hat() -> (OcrRecognizer, String) {
    for path in checkpoint_candidates() {
        if !path.is_file() {
            continue;
        }
        match Recognizer::load(&path) {
            Ok(rec) => {
                return (
                    OcrRecognizer(OcrEngine::Hat(rec)),
                    path.display().to_string(),
                );
            }
            Err(err) => eprintln!("ocr: skipped {}: {err}", path.display()),
        }
    }
    let rec = match Recognizer::from_bytes(CHECKPOINT) {
        Ok(rec) => rec,
        Err(err) => {
            // Embedded `model.mpk` is the previous BiLSTM checkpoint; HAT
            // is a different Module layout and will not load it.
            eprintln!(
                "ocr: embedded checkpoint incompatible with HAT ({err}); using random weights until a HAT checkpoint is trained"
            );
            return (
                OcrRecognizer(OcrEngine::Hat(
                    Recognizer::random().expect("failed to build random HAT recognizer"),
                )),
                "random (HAT, no trained checkpoint yet)".to_string(),
            );
        }
    };
    (
        OcrRecognizer(OcrEngine::Hat(rec)),
        "embedded checkpoints/model.mpk".to_string(),
    )
}

fn load_vlm_or_explain() -> (OcrRecognizer, String) {
    #[cfg(feature = "vlm")]
    {
        let Some(dir) = vlm_model_dir() else {
            panic!(
                "HWR_OCR_BACKEND=vlm but no HunyuanOCR checkpoint found.\n\
                 Download it next to the workspace, then rerun:\n\
                   huggingface-cli download tencent/HunyuanOCR \\\n\
                     --include config.json preprocessor_config.json tokenizer.json generation_config.json model.safetensors \\\n\
                     --include 'dflash/config.json' 'dflash/model.safetensors' \\\n\
                     --exclude 'v1.0/*' --exclude 'assets/*' \\\n\
                     --local-dir models/HunyuanOCR\n\
                 Or set HWR_VL_MODEL_DIR to that directory."
            );
        };
        let device = std::env::var("HWR_VL_DEVICE").unwrap_or_else(|_| "metal".to_string());
        match crate::vlm::VlmOcr::load(&dir, device.trim()) {
            Ok(vlm) => {
                let source = vlm.label(&dir, device.trim());
                return (OcrRecognizer(OcrEngine::Vlm(vlm)), source);
            }
            Err(err) => panic!("failed to load HunyuanOCR from {}: {err}", dir.display()),
        }
    }
    #[cfg(not(feature = "vlm"))]
    {
        eprintln!(
            "ocr: HWR_OCR_BACKEND requests a VLM, but this binary was built without `--features vlm`; using HAT"
        );
        load_hat()
    }
}

#[cfg(feature = "vlm")]
fn vlm_model_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HWR_VL_MODEL_DIR") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed);
            if looks_like_vlm_dir(&path) {
                return Some(path);
            }
            eprintln!(
                "ocr: HWR_VL_MODEL_DIR={} is not a HunyuanOCR directory (need config.json + tokenizer.json + safetensors)",
                path.display()
            );
        }
    }
    let root = workspace_root();
    for name in ["models/HunyuanOCR", "models/HunyuanOCR-1.5"] {
        let candidate = root.join(name);
        if looks_like_vlm_dir(&candidate) {
            return Some(candidate);
        }
    }
    None
}

#[cfg(feature = "vlm")]
fn looks_like_vlm_dir(path: &std::path::Path) -> bool {
    path.join("config.json").is_file()
        && path.join("preprocessor_config.json").is_file()
        && path.join("tokenizer.json").is_file()
        && (path.join("model.safetensors").is_file()
            || path.join("model.safetensors.index.json").is_file())
}

/// Walk up from cwd looking for the workspace (has both `checkpoints/` and
/// `hwr-app/`), so `cargo run -p hwr-app` works from the crate dir too.
fn workspace_root() -> PathBuf {
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for _ in 0..8 {
        if dir.join("checkpoints").is_dir() && dir.join("hwr-app").is_dir() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// `HWR_CHECKPOINT` wins, then `checkpoints/active.mpk` (symlink we keep
/// pointed at the current curriculum `--out`), then the known PBT artifacts
/// and the shipped `model.mpk`.
fn checkpoint_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(p) = std::env::var("HWR_CHECKPOINT") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            paths.push(PathBuf::from(trimmed));
        }
    }
    let root = workspace_root();
    for name in [
        "checkpoints/active.mpk",
        "checkpoints/pbt_digits.mpk",
        "checkpoints/pbt_alnum.mpk",
        "checkpoints/pbt_fullvocab.mpk",
        "checkpoints/model.mpk",
    ] {
        let p = root.join(name);
        if !paths.iter().any(|e| e == &p) {
            paths.push(p);
        }
    }
    paths
}
