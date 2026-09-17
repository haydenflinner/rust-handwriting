//! The recognizer, loaded once at startup from the pretrained checkpoint
//! or (with `--features vlm`) from a local HunyuanOCR directory.
//!
//! Inference runs on a dedicated worker thread so Hunyuan (or HAT) cannot
//! stall Bevy's input/render loop. Only [`Ink`] and the decoded string cross
//! the channel; the model never leaves the worker.
//!
//! Hunyuan and Bevy share one Metal GPU, so they take turns via a GPU token.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use bevy::prelude::*;
use hwr_ink::ink::Ink;
use hwr_model::Recognizer;

use crate::gpu_gate::GpuGate;

/// Pretrained on armrest's bundled `data/inks/*.txt` corpus — see
/// `hwr-model/src/bin/train.rs`. Embedded so a packaged app doesn't depend
/// on a runtime file path (matching how armrest embeds `english_ascii.tflite`).
/// Burn's Named-MessagePack format (see `train::save`) — not the old
/// candle-era `model.safetensors`, stale since the Burn migration.
///
/// Development runs prefer a file on disk (see [`checkpoint_candidates`])
/// so `hwr-app` can pick up the current PBT `--out` without a rebuild.
static CHECKPOINT: &[u8] = include_bytes!("../../checkpoints/model.mpk");

struct OcrEngine {
    inner: OcrBackend,
}

enum OcrBackend {
    Hat(Recognizer),
    #[cfg(feature = "vlm")]
    Vlm(crate::vlm::VlmOcr),
}

impl OcrEngine {
    fn recognize(&self, ink: &Ink, writing: &AtomicBool, gpu: &GpuGate) -> Result<String, String> {
        match &self.inner {
            OcrBackend::Hat(rec) => {
                gpu.ocr_acquire(writing);
                let _hold = gpu.ocr_hold();
                rec.recognize_greedy(ink).map_err(|err| err.to_string())
            }
            #[cfg(feature = "vlm")]
            OcrBackend::Vlm(vlm) => vlm.recognize(ink, writing, gpu),
        }
    }

    fn is_vlm(&self) -> bool {
        match &self.inner {
            OcrBackend::Hat(_) => false,
            #[cfg(feature = "vlm")]
            OcrBackend::Vlm(_) => true,
        }
    }
}

/// Main-thread handle to the OCR worker. `NonSend` because `mpsc::Receiver`
/// isn't `Sync`; we only poll it from the Bevy main thread anyway.
pub struct OcrClient {
    req_tx: Sender<(u64, Ink)>,
    res_rx: Receiver<(u64, Result<String, String>)>,
    next_id: u64,
    inflight: Option<u64>,
    is_vlm: bool,
}

impl OcrClient {
    pub fn is_vlm(&self) -> bool {
        self.is_vlm
    }

    pub fn submit(&mut self, ink: Ink) {
        self.next_id += 1;
        let id = self.next_id;
        self.inflight = Some(id);
        if self.req_tx.send((id, ink)).is_err() {
            eprintln!("ocr: worker thread is gone");
            self.inflight = None;
        }
    }

    /// Apply a completed job if it is still the latest submit. Stale results
    /// (superseded strokes, or a clear while inference was running) are dropped.
    pub fn poll(&mut self) -> Option<Result<String, String>> {
        loop {
            match self.res_rx.try_recv() {
                Ok((id, result)) => {
                    if self.inflight == Some(id) {
                        self.inflight = None;
                        return Some(result);
                    }
                }
                Err(_) => return None,
            }
        }
    }

    pub fn cancel(&mut self) {
        self.next_id += 1;
        self.inflight = None;
    }
}

/// Shared with the OCR worker: true while a stroke is in progress so Metal
/// inference can park and let Bevy keep the GPU.
#[derive(Resource, Clone)]
pub struct UiPointerDown(pub Arc<AtomicBool>);

impl UiPointerDown {
    pub fn set(&self, down: bool) {
        self.0.store(down, Ordering::Release);
    }
}

/// Path or label of the weights that were loaded, shown in test mode so
/// it's obvious which experiment the app is scoring against.
#[derive(Resource)]
pub struct OcrCheckpointSource(pub String);

pub struct OcrPlugin;

impl Plugin for OcrPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(UiPointerDown(Arc::new(AtomicBool::new(false))));
        // PreStartup so test-mode UI can read `OcrCheckpointSource` on Startup.
        app.add_systems(PreStartup, setup_recognizer);
    }
}

fn setup_recognizer(world: &mut World) {
    let (req_tx, req_rx) = mpsc::channel::<(u64, Ink)>();
    let (res_tx, res_rx) = mpsc::channel::<(u64, Result<String, String>)>();
    let (ready_tx, ready_rx) = mpsc::sync_channel::<(String, bool)>(1);
    let writing = world.resource::<UiPointerDown>().0.clone();
    let gpu = world.resource::<GpuGate>().clone();

    thread::Builder::new()
        .name("hwr-ocr".into())
        .spawn(move || {
            demote_worker_qos();
            gpu.ocr_acquire(&writing);
            let (engine, source) = load_recognizer();
            gpu.ocr_release();
            let is_vlm = engine.is_vlm();
            if ready_tx.send((source, is_vlm)).is_err() {
                return;
            }
            ocr_worker(engine, req_rx, res_tx, writing, gpu);
        })
        .expect("failed to spawn OCR worker thread");

    let (source, is_vlm) = ready_rx
        .recv()
        .expect("OCR worker died while loading the recognizer");
    eprintln!("ocr: loaded {source}");
    world.insert_resource(OcrCheckpointSource(source));
    world.insert_non_send(OcrClient {
        req_tx,
        res_rx,
        next_id: 0,
        inflight: None,
        is_vlm,
    });
}

fn ocr_worker(
    engine: OcrEngine,
    req_rx: Receiver<(u64, Ink)>,
    res_tx: Sender<(u64, Result<String, String>)>,
    writing: Arc<AtomicBool>,
    gpu: GpuGate,
) {
    loop {
        let (mut id, mut ink) = match req_rx.recv() {
            Ok(job) => job,
            Err(_) => break,
        };
        // Keep only the newest queued stroke so a word written while a
        // previous Hunyuan call is still running doesn't back up.
        while let Ok((next_id, next_ink)) = req_rx.try_recv() {
            id = next_id;
            ink = next_ink;
        }
        let result = engine.recognize(&ink, &writing, &gpu);
        if res_tx.send((id, result)).is_err() {
            break;
        }
    }
}

/// Drop below Bevy's UI thread so macOS prefers interactive cores / GPU for
/// sampling and drawing. Candle's rayon pool still self-promotes; this at
/// least keeps the Metal submit thread from matching the window's QoS.
fn demote_worker_qos() {
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        const QOS_CLASS_UTILITY: u32 = 0x11;
        pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
    }
}

fn requested_backend() -> String {
    std::env::var("HWR_OCR_BACKEND")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
}

fn load_recognizer() -> (OcrEngine, String) {
    let backend = requested_backend();
    match backend.as_str() {
        "vlm" | "hunyuan" | "hunyuanocr" | "paddle" | "paddleocr-vl" | "paddleocr_vl" => {
            load_vlm_or_explain()
        }
        "hat" | "ctc" => load_hat(),
        "" => {
            // Hunyuan is the default app backend when the vlm feature is on
            // and the checkpoint is on disk.
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

fn load_hat() -> (OcrEngine, String) {
    for path in checkpoint_candidates() {
        if !path.is_file() {
            continue;
        }
        match Recognizer::load(&path) {
            Ok(rec) => {
                return (
                    OcrEngine {
                        inner: OcrBackend::Hat(rec),
                    },
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
                OcrEngine {
                    inner: OcrBackend::Hat(
                        Recognizer::random().expect("failed to build random HAT recognizer"),
                    ),
                },
                "random (HAT, no trained checkpoint yet)".to_string(),
            );
        }
    };
    (
        OcrEngine {
            inner: OcrBackend::Hat(rec),
        },
        "embedded checkpoints/model.mpk".to_string(),
    )
}

fn load_vlm_or_explain() -> (OcrEngine, String) {
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
                return (
                    OcrEngine {
                        inner: OcrBackend::Vlm(vlm),
                    },
                    source,
                );
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
