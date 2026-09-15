pub mod augment;
pub mod corpus;
pub mod decode;
pub mod eval;
pub mod fused_lstm;
mod fused_lstm_kernel;
pub mod model;
pub mod probe;
pub mod recognizer;
pub mod spline;
pub mod template_match;
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
pub type Backend = burn::backend::Wgpu;
pub type TrainBackend = burn::backend::Autodiff<Backend>;
