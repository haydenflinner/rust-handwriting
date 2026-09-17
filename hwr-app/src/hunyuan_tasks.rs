//! Official HunyuanOCR-1.5 `--task-type` prompts.
//!
//! Tencent's client only exposes a task key (not free-form `--prompt`) so
//! users cannot silently degrade quality by rewriting the instruction.
//! Chinese wording is the one they recommend; English paper glosses are
//! less stable.

/// One of the 12 inference-time prompt variants.
pub struct HunyuanTask {
    pub id: &'static str,
    pub blurb: &'static str,
    pub prompt: &'static str,
}

pub const DEFAULT_TASK_ID: &str = "structured_parse";

/// Ordered for the test-mode dropdown: the two that already work, then the
/// official mixed text+formula line, then the rest.
pub const TASKS: &[HunyuanTask] = &[
    HunyuanTask {
        id: "structured_parse",
        blurb: "Scene / handwriting OCR",
        prompt: "提取图中的文字。",
    },
    HunyuanTask {
        id: "formula",
        blurb: "Formula → LaTeX",
        prompt: "识别图片中的公式，用LaTeX格式表示。",
    },
    HunyuanTask {
        id: "trans_other2en",
        blurb: "Extract + LaTeX, then English",
        prompt: "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，再将文字内容翻译为英文。",
    },
    HunyuanTask {
        id: "doc_parse",
        blurb: "Document → markdown",
        prompt: "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。",
    },
    HunyuanTask {
        id: "layout_parse",
        blurb: "Layout + full parse",
        prompt: "提取文档图片中所有内容用markdown格式表示，表格用html格式表达，文档中公式用latex格式表示，请按照阅读顺序组织进行全文解析，并输出版式分析信息。",
    },
    HunyuanTask {
        id: "spotting_hunyuan",
        blurb: "Detect + recognize (Hunyuan boxes)",
        prompt: "检测并识别图片中的文字，将文本坐标格式化输出。",
    },
    HunyuanTask {
        id: "spotting_json",
        blurb: "Detect + recognize (JSON boxes)",
        prompt: "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 输出格式为 JSON 数组，每个元素必须包含：\"box\": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；\"text\": \"识别出的文字内容\"。 注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。",
    },
    HunyuanTask {
        id: "layout",
        blurb: "Layout only",
        prompt: "按照阅读顺序解析图中的版式信息。",
    },
    HunyuanTask {
        id: "chart_parse",
        blurb: "Chart → Mermaid/Markdown",
        prompt: "解析图中的图表，对于流程图使用Mermaid格式表示，其他图表使用Markdown格式表示。",
    },
    HunyuanTask {
        id: "table",
        blurb: "Table → HTML",
        prompt: "把图中的表格解析为HTML。",
    },
    HunyuanTask {
        id: "doc_trans_en2zh",
        blurb: "Document EN → ZH",
        prompt: "先解析文档，再将文档内容翻译为中文，其中页眉、页脚忽略，公式用latex格式表示，表格用html格式表示。",
    },
    HunyuanTask {
        id: "trans_other2zh",
        blurb: "Extract + LaTeX, then Chinese",
        prompt: "按照阅读顺序，提取图中文字，公式用latex格式表示，表格用markdown格式表示，再将文字内容翻译为中文。",
    },
];

pub fn task_by_id(id: &str) -> Option<&'static HunyuanTask> {
    TASKS.iter().find(|task| task.id.eq_ignore_ascii_case(id))
}

pub fn prompt_for(id: &str) -> Option<&'static str> {
    task_by_id(id).map(|task| task.prompt)
}

/// `HWR_VL_TASK` if it names a known key, otherwise [`DEFAULT_TASK_ID`].
pub fn initial_task_id() -> &'static str {
    let env = std::env::var("HWR_VL_TASK").ok();
    let id = env.as_deref().map(str::trim);
    if let Some(id) = id {
        if let Some(task) = task_by_id(id) {
            return task.id;
        }
        eprintln!("ocr: unknown HWR_VL_TASK={id:?}; using {DEFAULT_TASK_ID}");
    }
    DEFAULT_TASK_ID
}
