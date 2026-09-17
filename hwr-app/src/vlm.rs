//! HunyuanOCR backend: rasterize ink, then transcribe handwriting.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use hwr_ink::ink::Ink;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::HunyuanOcr;

use crate::gpu_gate::{wait_while_writing, GpuGate};
use crate::ink_image::rasterize_ink_rgb;

/// Short crop of stylus ink, not a full document page. Hunyuan's default
/// spotting prompt asks for boxes; we want plain text.
const HANDWRITING_PROMPT: &str = "Transcribe the handwritten English text in this image exactly as written. Output only the text. Do not add bounding boxes, markdown, labels, or commentary.";

pub struct VlmOcr {
    model: HunyuanOcr,
}

impl VlmOcr {
    pub fn load(model_dir: impl AsRef<Path>, device: &str) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        // Candle's Metal path is fastest in f16 on Apple Silicon; Auto would
        // prefer bf16 when the probe succeeds, which is slower here.
        // SAFETY: called once on the OCR worker at load, before generate
        // and before Bevy starts extra work on this thread.
        if std::env::var_os("OAR_VL_DTYPE").is_none() {
            unsafe {
                std::env::set_var("OAR_VL_DTYPE", "f16");
            }
        }
        let device = parse_device(device).map_err(|err| err.to_string())?;
        let want_dflash = dflash_requested() && looks_like_dflash_dir(&model_dir.join("dflash"));
        eprintln!(
            "ocr: loading HunyuanOCR from {} on {device:?}{}",
            model_dir.display(),
            if want_dflash { " (DFlash)" } else { "" }
        );
        let start = Instant::now();
        let model = if want_dflash {
            match HunyuanOcr::from_dir_with_dflash(model_dir, device.clone()) {
                Ok(model) => model,
                Err(err) => {
                    eprintln!("ocr: DFlash load failed ({err}); loading HunyuanOCR without it");
                    HunyuanOcr::from_dir(model_dir, device).map_err(|err| err.to_string())?
                }
            }
        } else {
            HunyuanOcr::from_dir(model_dir, device).map_err(|err| err.to_string())?
        };
        eprintln!(
            "ocr: HunyuanOCR {} loaded in {:.1}s{}",
            model.version(),
            start.elapsed().as_secs_f32(),
            model
                .dflash_num_speculative_tokens()
                .map(|n| format!(", DFlash {n} draft tokens"))
                .unwrap_or_default()
        );
        Ok(Self { model })
    }

    pub fn label(&self, model_dir: &Path, device: &str) -> String {
        let dflash = if self.model.dflash_enabled() {
            ", DFlash"
        } else {
            ""
        };
        format!(
            "HunyuanOCR {} ({}, {}{dflash})",
            self.model.version(),
            model_dir.display(),
            device
        )
    }

    pub fn recognize(
        &self,
        ink: &Ink,
        writing: &AtomicBool,
        gpu: &GpuGate,
    ) -> Result<String, String> {
        wait_while_writing(writing);
        let Some(image) = rasterize_ink_rgb(ink) else {
            return Ok(String::new());
        };
        if let Ok(path) = std::env::var("HWR_VL_DUMP_PNG") {
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                if let Err(err) = image.save(trimmed) {
                    eprintln!("ocr: failed to dump raster to {trimmed}: {err}");
                } else {
                    eprintln!("ocr: dumped raster to {trimmed}");
                }
            }
        }
        let prompt =
            std::env::var("HWR_VL_PROMPT").unwrap_or_else(|_| HANDWRITING_PROMPT.to_string());
        let start = Instant::now();
        gpu.ocr_acquire(writing);
        let _hold = gpu.ocr_hold();
        let outputs = self
            .model
            .generate_with_step(&[image], &[prompt.as_str()], 256, || {
                // Drain this layer's Metal work while we still hold the GPU
                // token, then let Bevy render before the next layer/token.
                let _ = self.model.device().synchronize();
                gpu.ocr_release();
                wait_while_writing(writing);
                gpu.ocr_acquire(writing);
            })
            .map_err(|err| err.to_string())?;
        let text = outputs
            .into_iter()
            .next()
            .ok_or_else(|| "HunyuanOCR returned no result".to_string())?
            .map_err(|err| err.to_string())?;
        let text = text.trim().to_string();
        eprintln!(
            "ocr: HunyuanOCR inferred in {:.1}s → {text:?}",
            start.elapsed().as_secs_f32()
        );
        Ok(text)
    }
}

fn dflash_requested() -> bool {
    !matches!(
        std::env::var("HWR_VL_NO_DFLASH").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE")
    )
}

fn looks_like_dflash_dir(path: &Path) -> bool {
    path.join("config.json").is_file() && path.join("model.safetensors").is_file()
}
