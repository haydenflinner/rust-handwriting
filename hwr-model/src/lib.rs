pub mod augment;
pub mod corpus;
pub mod decode;
pub mod eval;
#[cfg(not(target_arch = "wasm32"))]
pub mod fused_lstm;
// Not compiled: `fused_lstm_kernel.rs` launches against a raw CubeTensor.
// HAT needs burn-wgpu `fusion` (`FusionTensor`), which that kernel cannot
// unwrap. Source stays in-tree for a no-fusion LSTM revival.
// mod fused_lstm_kernel;
pub mod model;
pub mod onnet;
pub mod pbt;
#[cfg(not(target_arch = "wasm32"))]
pub mod probe;
pub mod recognizer;
pub mod spline;
pub mod template_match;
#[cfg(not(target_arch = "wasm32"))]
pub mod train;

pub use decode::{Beam, Greedy, LanguageModel, ModelOutput, RawOutput};
pub use recognizer::{Error, Recognizer};

/// The backend used for inference (and, wrapped in `Autodiff`, for
/// training). Wgpu targets Metal on macOS via CubeCL — see the module docs
/// on `train` for why this replaced candle: candle has no native CTC op,
/// so our hand-rolled CTC forward-backward had to run as plain CPU Rust,
/// forcing a GPU<->CPU round-trip every training step. Burn's `CTCLoss`
/// dispatches to a per-backend kernel (verified against PyTorch's own CTC
/// output/gradients — see `burn-nn`'s `pytorch_comparison_tests`), so the
/// whole loss computation stays resident on whatever device `Backend` is.
///
/// On wasm this is the same Wgpu type, backed by WebGPU — call
/// [`init_wgpu`] before constructing a [`Recognizer`].
pub type Backend = burn::backend::Wgpu;
#[cfg(not(target_arch = "wasm32"))]
pub type TrainBackend = burn::backend::Autodiff<Backend>;

/// Browser WebGPU must be initialized asynchronously before any tensor ops.
/// Native `WgpuDevice::default()` does this synchronously on first use.
#[cfg(target_arch = "wasm32")]
pub async fn init_wgpu() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let _ = burn::backend::wgpu::init_setup_async::<
        burn::backend::wgpu::graphics::WebGpu,
    >(&device, Default::default())
    .await;
}
