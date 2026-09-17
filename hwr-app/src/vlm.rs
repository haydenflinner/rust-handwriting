//! HunyuanOCR backend: rasterize ink, then transcribe handwriting.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use hwr_ink::ink::Ink;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::HunyuanOcr;

use crate::gpu_gate::{wait_while_writing, GpuGate};
use crate::ink_image::rasterize_ink_rgb;

/// Official HunyuanOCR-1.5 task prompts (Chinese). Tencent's client only
/// exposes `--task-type`; they recommend the Chinese wording for quality.
/// Override with `HWR_VL_TASK` (key below) or `HWR_VL_PROMPT` (raw string).
///
/// `structured_parse` is the general-scene OCR line (street view / ink).
/// `formula` is the one that should turn a 2×2 matrix into LaTeX.
fn official_task_prompt(task: &str) -> Option<&'static str> {
    Some(match task {
        "doc_parse" => {
            "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
        }
        "structured_parse" => "提取图中的文字。",
        "spotting_json" => {
            "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 输出格式为 JSON 数组，每个元素必须包含：\"box\": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；\"text\": \"识别出的文字内容\"。 注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。"
        }
        "spotting_hunyuan" => "检测并识别图片中的文字，将文本坐标格式化输出。",
        "layout" => "按照阅读顺序解析图中的版式信息。",
        "layout_parse" => {
            "提取文档图片中所有内容用markdown格式表示，表格用html格式表达，文档中公式用latex格式表示，请按照阅读顺序组织进行全文解析，并输出版式分析信息。"
        }
        "chart_parse" => {
            "解析图中的图表，对于流程图使用Mermaid格式表示，其他图表使用Markdown格式表示。"
        }
        "formula" => "识别图片中的公式，用LaTeX格式表示。",
        "table" => "把图中的表格解析为HTML。",
        "doc_trans_en2zh" => {
            "先解析文档，再将文档内容翻译为中文，其中页眉、页脚忽略，公式用latex格式表示，表格用html格式表示。"
        }
        "trans_other2en" => {
            "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，再将文字内容翻译为英文。"
        }
        "trans_other2zh" => {
            "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，再将文字内容翻译为中文。"
        }
        _ => return None,
    })
}

fn hunyuan_instruction() -> String {
    if let Ok(raw) = std::env::var("HWR_VL_PROMPT") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    let task = std::env::var("HWR_VL_TASK").unwrap_or_else(|_| "structured_parse".to_string());
    let task = task.trim().to_ascii_lowercase();
    official_task_prompt(&task)
        .unwrap_or_else(|| {
            eprintln!("ocr: unknown HWR_VL_TASK={task:?}; using structured_parse");
            official_task_prompt("structured_parse").expect("structured_parse is a known task")
        })
        .to_string()
}

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
        let prompt = hunyuan_instruction();
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
