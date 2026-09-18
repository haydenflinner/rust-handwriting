mod calibrate;
mod gpu_gate;
mod hunyuan_tasks;
#[cfg(feature = "vlm")]
mod ink_image;
mod mode;
mod ocr;
mod prompts;
mod review;
mod storage;
mod test_mode;
mod typst_convert;
mod typst_preview;
mod ui_theme;
#[cfg(feature = "vlm")]
mod vlm;
#[cfg(all(feature = "vlm", not(target_arch = "wasm32")))]
mod web_server;
mod writing_cell;

use bevy::prelude::*;

const BACKGROUND: Color = Color::srgb(0.08, 0.08, 0.1);

fn main() {
    #[cfg(all(feature = "vlm", not(target_arch = "wasm32")))]
    if std::env::args().any(|arg| arg == "--web") {
        web_server::run();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        wasm_bindgen_futures::spawn_local(async {
            match start_web().await {
                Ok(()) => {}
                Err(err) => log(&format!("hwr: failed to start: {err}")),
            }
        });
    }
    #[cfg(not(target_arch = "wasm32"))]
    start_app(None);
}

#[cfg(target_arch = "wasm32")]
async fn start_web() -> Result<(), String> {
    log("hwr: starting Bevy…");
    let checkpoint = fetch_bytes("model.mpk").await;
    match &checkpoint {
        Some(bytes) => log(&format!("hwr: loaded model.mpk ({} bytes)", bytes.len())),
        None => log("hwr: model.mpk missing; recognition will use random weights"),
    }
    start_app(checkpoint);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
async fn fetch_bytes(url: &str) -> Option<Vec<u8>> {
    use wasm_bindgen::JsCast;
    let window = web_sys::window()?;
    let resp_value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url))
        .await
        .ok()?;
    let resp: web_sys::Response = resp_value.dyn_into().ok()?;
    if !resp.ok() {
        log(&format!(
            "hwr: fetch {url} failed with status {}",
            resp.status()
        ));
        return None;
    }
    let buf = wasm_bindgen_futures::JsFuture::from(resp.array_buffer().ok()?)
        .await
        .ok()?;
    Some(js_sys::Uint8Array::new(&buf).to_vec())
}

fn start_app(web_checkpoint: Option<Vec<u8>>) {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "hwr".into(),
            fit_canvas_to_parent: true,
            ..default()
        }),
        ..default()
    }))
    .insert_resource(ClearColor(BACKGROUND));
    #[cfg(target_arch = "wasm32")]
    app.insert_resource(ocr::WasmCheckpoint(web_checkpoint));
    #[cfg(not(target_arch = "wasm32"))]
    let _ = web_checkpoint;
    app.add_plugins((
        gpu_gate::GpuGatePlugin,
        writing_cell::WritingCellPlugin,
        ocr::OcrPlugin,
        mode::ModePlugin,
        test_mode::TestModePlugin,
        calibrate::CalibratePlugin,
        review::ReviewPlugin,
    ))
    .add_systems(Startup, setup_camera)
    .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

pub(crate) fn log(msg: &str) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&msg.into());
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{msg}");
}
