//! Exclusive GPU token shared by Bevy's render thread and the OCR worker.
//!
//! Hunyuan (Candle/Metal) and Bevy (wgpu/Metal) cannot submit to the same
//! Apple GPU at once: overlapping command buffers hitch the swapchain and
//! showed up as pink blank frames. This gate makes them take turns.
//!
//! A kernel already dispatched cannot be cancelled — the GPU runs it to
//! completion. The token only decides who may *start* the next chunk.

use std::sync::atomic::AtomicBool;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use bevy::prelude::*;
use bevy::render::{Render, RenderApp, RenderSystems};

/// Park while the user is writing so Hunyuan does not occupy the GPU
/// for the duration of a stroke.
#[cfg(not(target_arch = "wasm32"))]
pub fn wait_while_writing(writing: &AtomicBool) {
    while writing.load(Ordering::Acquire) {
        thread::sleep(Duration::from_millis(4));
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GpuOwner {
    Free,
    Bevy,
    Ocr,
}

struct GpuState {
    owner: GpuOwner,
    /// Bevy has entered acquire and is waiting or about to mark itself owner.
    /// OCR must not steal `Free` in that window.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    bevy_waiting: u32,
}

/// Cloneable token. Inserted in both the main world and the [`RenderApp`].
#[derive(Resource, Clone)]
pub struct GpuGate(Arc<GpuGateInner>);

struct GpuGateInner {
    state: Mutex<GpuState>,
    cv: Condvar,
}

impl GpuGate {
    pub fn new() -> Self {
        Self(Arc::new(GpuGateInner {
            state: Mutex::new(GpuState {
                owner: GpuOwner::Free,
                bevy_waiting: 0,
            }),
            cv: Condvar::new(),
        }))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GpuState> {
        self.0.state.lock().unwrap_or_else(|err| err.into_inner())
    }

    /// Take the GPU for this Bevy frame, or fail immediately if OCR is in a
    /// kernel. Never park the render thread: pipelined extract waits on it,
    /// so a condvar stall here froze input while Hunyuan ran.
    ///
    /// A failed try still sets `bevy_waiting` so OCR yields after the kernel
    /// instead of starving the swapchain for the whole job.
    #[cfg(not(target_arch = "wasm32"))]
    fn bevy_try_acquire(&self) -> bool {
        let mut state = self.lock();
        if state.owner == GpuOwner::Ocr {
            if state.bevy_waiting == 0 {
                state.bevy_waiting = 1;
                self.0.cv.notify_all();
            }
            return false;
        }
        state.owner = GpuOwner::Bevy;
        state.bevy_waiting = 0;
        self.0.cv.notify_all();
        true
    }

    fn bevy_release(&self) {
        let mut state = self.lock();
        if state.owner == GpuOwner::Bevy {
            state.owner = GpuOwner::Free;
            self.0.cv.notify_all();
        }
    }

    pub fn ocr_acquire(&self, writing: &AtomicBool) {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = writing;
            let mut state = self.lock();
            state.owner = GpuOwner::Ocr;
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        loop {
            wait_while_writing(writing);
            let mut state = self.lock();
            loop {
                if writing.load(Ordering::Acquire) {
                    break;
                }
                if state.owner == GpuOwner::Free && state.bevy_waiting == 0 {
                    state.owner = GpuOwner::Ocr;
                    self.0.cv.notify_all();
                    return;
                }
                let waited = self
                    .0
                    .cv
                    .wait_timeout(state, Duration::from_millis(4))
                    .unwrap_or_else(|err| err.into_inner());
                state = waited.0;
            }
        }
    }

    pub fn ocr_release(&self) {
        let mut state = self.lock();
        if state.owner == GpuOwner::Ocr {
            state.owner = GpuOwner::Free;
            self.0.cv.notify_all();
        }
    }

    /// Hold the token until drop. Safe to `ocr_release`/`ocr_acquire` inside
    /// the scope; drop still releases if OCR owns the GPU.
    pub fn ocr_hold(&self) -> OcrGpuHold<'_> {
        OcrGpuHold(self)
    }
}

pub struct OcrGpuHold<'a>(&'a GpuGate);

impl Drop for OcrGpuHold<'_> {
    fn drop(&mut self) {
        self.0.ocr_release();
    }
}

pub struct GpuGatePlugin;

impl Plugin for GpuGatePlugin {
    fn build(&self, app: &mut App) {
        let gate = GpuGate::new();
        app.insert_resource(gate.clone());
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.insert_resource(gate);
        render_app.init_resource::<SkipGpuFrame>();
        render_app.add_systems(First, gpu_bevy_acquire);
        // Skip wgpu submit/present while OCR holds Metal — overlapping
        // command buffers were the pink-blank hitch. The previous frame stays
        // on screen until Hunyuan yields.
        render_app.configure_sets(Render, RenderSystems::Render.run_if(not(skip_gpu_frame)));
        render_app.add_systems(
            Render,
            gpu_bevy_release
                .after(RenderSystems::Render)
                .run_if(not(skip_gpu_frame)),
        );
    }
}

#[derive(Resource, Default)]
struct SkipGpuFrame(bool);

fn skip_gpu_frame(skip: Res<SkipGpuFrame>) -> bool {
    skip.0
}

fn gpu_bevy_acquire(gate: Res<GpuGate>, mut skip: ResMut<SkipGpuFrame>) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = gate;
        skip.0 = false;
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        skip.0 = !gate.bevy_try_acquire();
    }
}

fn gpu_bevy_release(gate: Res<GpuGate>) {
    gate.bevy_release();
}
