//! Browser UI for native HunyuanOCR.
//!
//! Hunyuan is Candle/Metal and far too large for WASM, so the model stays in
//! this process and the page is just a drawing surface + result panel.

use std::io::Cursor;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use hwr_ink::ink::Ink;
use serde::Deserialize;
use tiny_http::{Header, Method, Response, Server, StatusCode};

use crate::gpu_gate::GpuGate;
use crate::hunyuan_tasks::{self, TASKS};
use crate::typst_convert::latex_to_typst;

const PAGE: &str = include_str!("hunyuan_web.html");

#[derive(Deserialize)]
struct RecognizeRequest {
    task: Option<String>,
    strokes: Vec<Vec<StrokePoint>>,
}

#[derive(Deserialize)]
struct StrokePoint {
    x: f32,
    y: f32,
    #[serde(default)]
    t: f32,
}

pub fn run() -> ! {
    let (vlm, source) = crate::ocr::load_vlm_ocr().unwrap_or_else(|err| {
        panic!("{err}");
    });
    let gpu = GpuGate::new();
    let writing = AtomicBool::new(false);
    let port: u16 = std::env::var("HWR_WEB_PORT")
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(4000);
    let addr = format!("127.0.0.1:{port}");
    let server = Server::http(&addr).unwrap_or_else(|err| {
        panic!("failed to bind {addr}: {err}");
    });
    eprintln!("hwr: HunyuanOCR web UI at http://{addr}/");
    eprintln!("hwr: {source}");
    let _ = std::process::Command::new("open")
        .arg(format!("http://{addr}/"))
        .status();

    for mut request in server.incoming_requests() {
        let method = request.method().clone();
        let url = request.url().to_string();
        let path = url.split('?').next().unwrap_or("/");
        let response = match (method, path) {
            (Method::Get, "/") => html_response(PAGE),
            (Method::Get, "/tasks") => json_response(200, tasks_json()),
            (Method::Post, "/recognize") => {
                let mut body = String::new();
                match request.as_reader().read_to_string(&mut body) {
                    Ok(_) => match recognize(&vlm, &gpu, &writing, &body) {
                        Ok(json) => json_response(200, json),
                        Err(err) => json_response(400, serde_json::json!({ "error": err })),
                    },
                    Err(err) => json_response(
                        400,
                        serde_json::json!({ "error": format!("read body: {err}") }),
                    ),
                }
            }
            _ => json_response(404, serde_json::json!({ "error": "not found" })),
        };
        if let Err(err) = request.respond(response) {
            eprintln!("hwr: failed to respond: {err}");
        }
    }
    unreachable!("tiny_http server ended");
}

fn recognize(
    vlm: &crate::vlm::VlmOcr,
    gpu: &GpuGate,
    writing: &AtomicBool,
    body: &str,
) -> Result<serde_json::Value, String> {
    let req: RecognizeRequest =
        serde_json::from_str(body).map_err(|err| format!("invalid JSON: {err}"))?;
    let ink = ink_from_strokes(&req.strokes);
    if ink.is_empty() {
        return Ok(serde_json::json!({
            "text": "",
            "typst": "",
            "ms": 0
        }));
    }
    let prompt = req
        .task
        .as_deref()
        .and_then(hunyuan_tasks::prompt_for)
        .map(str::to_string);
    let start = Instant::now();
    let text = vlm.recognize(&ink, writing, gpu, prompt.as_deref())?;
    let typst = latex_to_typst(&text);
    Ok(serde_json::json!({
        "text": text,
        "typst": typst,
        "ms": start.elapsed().as_millis() as u64
    }))
}

fn ink_from_strokes(strokes: &[Vec<StrokePoint>]) -> Ink {
    let mut ink = Ink::new();
    for stroke in strokes {
        if stroke.is_empty() {
            continue;
        }
        for point in stroke {
            ink.push(point.x, point.y, point.t);
        }
        ink.pen_up();
    }
    ink
}

fn tasks_json() -> serde_json::Value {
    serde_json::json!({
        "default": hunyuan_tasks::initial_task_id(),
        "tasks": TASKS.iter().map(|task| {
            serde_json::json!({
                "id": task.id,
                "blurb": task.blurb,
                "prompt_en": task.prompt_en
            })
        }).collect::<Vec<_>>()
    })
}

fn html_response(body: &'static str) -> Response<Cursor<Vec<u8>>> {
    let mut response = Response::from_string(body).with_status_code(StatusCode(200));
    add_header(&mut response, "Content-Type", "text/html; charset=utf-8");
    response
}

fn json_response(status: u16, value: serde_json::Value) -> Response<Cursor<Vec<u8>>> {
    let mut response = Response::from_string(value.to_string()).with_status_code(StatusCode(status));
    add_header(
        &mut response,
        "Content-Type",
        "application/json; charset=utf-8",
    );
    response
}

fn add_header(response: &mut Response<Cursor<Vec<u8>>>, name: &str, value: &str) {
    if let Ok(header) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
        response.add_header(header);
    }
}
