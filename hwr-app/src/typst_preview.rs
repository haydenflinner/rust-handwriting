//! Rasterize converted Typst (from mitex) to a PNG-like RGBA buffer.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, Sender};

use typst::diag::{FileError, FileResult, Warned};
use typst::foundations::{Bytes, Datetime};
use typst::layout::PagedDocument;
use typst::syntax::{FileId, Source, VirtualPath};
use typst::text::{Font, FontBook};
use typst::utils::LazyHash;
use typst::{Library, World};
use typst_kit::fonts::Fonts;

const MAX_EDGE: u32 = 4096;

struct PreviewJob {
    source: String,
    pixel_per_pt: f32,
}

pub struct PreviewFrame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub enum PreviewResult {
    Empty,
    Frame(PreviewFrame),
    Error(String),
}

/// Main-thread handle. Native builds compile on a worker so Typst cannot stall
/// the UI thread; wasm compiles on poll (no threads).
pub struct TypstPreviewClient {
    #[cfg(not(target_arch = "wasm32"))]
    req_tx: Sender<PreviewJob>,
    #[cfg(not(target_arch = "wasm32"))]
    res_rx: Receiver<PreviewResult>,
    #[cfg(target_arch = "wasm32")]
    fonts: Option<Fonts>,
    #[cfg(target_arch = "wasm32")]
    pending: Option<PreviewJob>,
    last_sent: String,
    last_ppp: f32,
}

impl TypstPreviewClient {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (req_tx, req_rx) = mpsc::channel::<PreviewJob>();
            let (res_tx, res_rx) = mpsc::channel::<PreviewResult>();
            std::thread::Builder::new()
                .name("hwr-typst".into())
                .spawn(move || preview_worker(req_rx, res_tx))
                .expect("failed to spawn Typst preview worker");
            Self {
                req_tx,
                res_rx,
                last_sent: String::new(),
                last_ppp: 0.0,
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self {
                fonts: None,
                pending: None,
                last_sent: String::new(),
                last_ppp: 0.0,
            }
        }
    }

    pub fn submit(&mut self, source: String, pixel_per_pt: f32) {
        let pixel_per_pt = pixel_per_pt.clamp(1.0, 8.0);
        if source == self.last_sent && (pixel_per_pt - self.last_ppp).abs() < 0.05 {
            return;
        }
        self.last_sent = source.clone();
        self.last_ppp = pixel_per_pt;
        let job = PreviewJob {
            source,
            pixel_per_pt,
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self.req_tx.send(job);
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.pending = Some(job);
        }
    }

    pub fn poll(&mut self) -> Option<PreviewResult> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.res_rx.try_recv().ok()
        }
        #[cfg(target_arch = "wasm32")]
        {
            let job = self.pending.take()?;
            let fonts = self.fonts.get_or_insert_with(load_fonts);
            Some(render_source(fonts, &job.source, job.pixel_per_pt))
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn preview_worker(req_rx: Receiver<PreviewJob>, res_tx: Sender<PreviewResult>) {
    let fonts = load_fonts();
    loop {
        let mut job = match req_rx.recv() {
            Ok(job) => job,
            Err(_) => break,
        };
        while let Ok(next) = req_rx.try_recv() {
            job = next;
        }
        if res_tx
            .send(render_source(&fonts, &job.source, job.pixel_per_pt))
            .is_err()
        {
            break;
        }
    }
}

fn load_fonts() -> Fonts {
    let mut searcher = Fonts::searcher();
    #[cfg(target_arch = "wasm32")]
    searcher.include_system_fonts(false);
    searcher.search()
}

fn render_source(fonts: &Fonts, source: &str, pixel_per_pt: f32) -> PreviewResult {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return PreviewResult::Empty;
    }
    if trimmed.starts_with("// mitex:") {
        return PreviewResult::Error(trimmed.trim_start_matches("// ").to_string());
    }

    let world = PreviewWorld::new(fonts, wrap_preview_document(trimmed));
    let Warned { output, warnings: _ } = typst::compile::<PagedDocument>(&world);
    match output {
        Ok(doc) => match rasterize(&doc, pixel_per_pt) {
            Ok(frame) => PreviewResult::Frame(frame),
            Err(err) => PreviewResult::Error(err),
        },
        Err(errors) => {
            let message = errors
                .iter()
                .map(|err| err.message.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            PreviewResult::Error(if message.is_empty() {
                "Typst failed to compile this snippet".to_string()
            } else {
                message
            })
        }
    }
}

fn wrap_preview_document(body: &str) -> String {
    format!(
        "#set page(width: auto, height: auto, margin: 12pt, fill: rgb(\"#f4f1ea\"))\n\
         #set text(size: 18pt, fill: rgb(\"#1b1b1b\"))\n\
         #set par(justify: false)\n\n\
         {body}\n"
    )
}

fn rasterize(doc: &PagedDocument, pixel_per_pt: f32) -> Result<PreviewFrame, String> {
    let page = doc
        .pages
        .first()
        .ok_or_else(|| "Typst produced no pages".to_string())?;
    let pixmap = typst_render::render(page, pixel_per_pt);
    let mut width = pixmap.width();
    let mut height = pixmap.height();
    if width == 0 || height == 0 {
        return Err("Typst produced an empty page".to_string());
    }
    let mut rgba = unpremultiply(pixmap.data());
    if width.max(height) > MAX_EDGE {
        // Keep the GPU texture bounded for a long equation.
        let scale = MAX_EDGE as f32 / width.max(height) as f32;
        width = ((width as f32) * scale).max(1.0).round() as u32;
        height = ((height as f32) * scale).max(1.0).round() as u32;
        rgba = nearest_resize(pixmap.data(), pixmap.width(), pixmap.height(), width, height);
        rgba = unpremultiply(&rgba);
    }
    Ok(PreviewFrame {
        width,
        height,
        rgba,
    })
}

fn unpremultiply(data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    for px in out.chunks_exact_mut(4) {
        let a = px[3] as u32;
        if a > 0 && a < 255 {
            px[0] = ((px[0] as u32 * 255) / a).min(255) as u8;
            px[1] = ((px[1] as u32 * 255) / a).min(255) as u8;
            px[2] = ((px[2] as u32 * 255) / a).min(255) as u8;
        }
    }
    out
}

fn nearest_resize(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0; (dw * dh * 4) as usize];
    for y in 0..dh {
        let sy = y * sh / dh;
        for x in 0..dw {
            let sx = x * sw / dw;
            let si = ((sy * sw + sx) * 4) as usize;
            let di = ((y * dw + x) * 4) as usize;
            out[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
    out
}

struct PreviewWorld<'a> {
    library: LazyHash<Library>,
    book: LazyHash<FontBook>,
    fonts: &'a Fonts,
    source: Source,
}

impl<'a> PreviewWorld<'a> {
    fn new(fonts: &'a Fonts, source: String) -> Self {
        let id = FileId::new(None, VirtualPath::new("preview.typ"));
        Self {
            library: LazyHash::new(Library::default()),
            book: LazyHash::new(fonts.book.clone()),
            fonts,
            source: Source::new(id, source),
        }
    }
}

impl World for PreviewWorld<'_> {
    fn library(&self) -> &LazyHash<Library> {
        &self.library
    }

    fn book(&self) -> &LazyHash<FontBook> {
        &self.book
    }

    fn main(&self) -> FileId {
        self.source.id()
    }

    fn source(&self, id: FileId) -> FileResult<Source> {
        if id == self.source.id() {
            Ok(self.source.clone())
        } else {
            Err(FileError::NotFound(id.vpath().as_rooted_path().into()))
        }
    }

    fn file(&self, id: FileId) -> FileResult<Bytes> {
        Err(FileError::NotFound(id.vpath().as_rooted_path().into()))
    }

    fn font(&self, index: usize) -> Option<Font> {
        self.fonts.fonts.get(index)?.get()
    }

    fn today(&self, _offset: Option<i64>) -> Option<Datetime> {
        None
    }
}
