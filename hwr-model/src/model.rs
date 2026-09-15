//! The recognition network: a stack of bidirectional LSTM layers followed by
//! a per-step dense+softmax classifier, matching the architecture trained by
//! armrest's `script/training.py` (`build_model`): 5 layers, hidden size 64
//! per direction, `concat` merge mode, CTC-friendly softmax output.
//!
//! Built on our own `FusedBiLstmLayer` (see `crate::fused_lstm` for why:
//! stock `burn::nn::BiLstm`'s per-timestep Rust loop makes Burn's autodiff
//! track O(sequence length) graph nodes per layer, which we measured
//! scaling super-linearly and dominating training time; `FusedBiLstmLayer`
//! registers one custom-gradient node per direction instead), `BatchNorm`
//! after each layer, and `Dropout`. Both `BatchNorm` and `Dropout`
//! automatically become no-ops when `B` isn't an autodiff backend (see
//! `Recognizer::forward_logits`) — so a plain `B::InnerBackend` obtained via
//! `AutodiffModule::valid()` gives correct, deterministic inference for
//! free, no `train: bool` threading.
//!
//! `BatchNorm` is here for optimization, not regularization (that's
//! `Dropout`'s job): normalizing each layer's output keeps activation scale
//! consistent through the 5-layer stack, which a reference LSTM-CTC-OCR
//! implementation (see `lstm-ctc-ocr/3_phonernn.lua`) pairs with dropout for
//! exactly this reason. Without it, we saw the model get stuck predicting
//! blank at every timestep (a well-documented CTC local optimum) far longer
//! than expected even on a trivially small overfit set — and even with it,
//! the same collapse persisted across a wide hyperparameter sweep (LR,
//! optimizer, batch size, dropout on/off), all on plain raw-delta input.
//!
//! `tcn` is a small front-end (`Config::tcn_channels`, `None` to disable)
//! of dilated 1D convolutions ahead of the BiLSTM stack: a couple of
//! `Conv1d` + ReLU layers that look at a local window of nearby timesteps
//! directly (dilation doubling per layer for exponentially-growing reach —
//! standard TCN recipe, Bai et al. 2018), extracting local stroke features
//! (corners, direction reversals) before the recurrent stack has to do that
//! *and* long-range sequence modeling at every one of its 5 layers. Tried
//! specifically because the collapse looked like an optimization-dynamics
//! problem more than a representational one (stock `BiLstm` and our fused
//! kernel hit the identical wall), and a front-end that gives early layers
//! a more direct gradient path is a plausible, bounded way to test that
//! without a much larger architecture rewrite (e.g. a full transformer).

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig1d,
};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::decode;
use crate::fused_lstm::{FusedBiLstmLayer, FusedLstm};

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub input_width: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub classes: usize,
    /// Output channels of the TCN front-end's conv layers (see
    /// `Recognizer`'s docs for why it exists). `None` disables the
    /// front-end entirely, feeding `input_width`-wide raw deltas straight
    /// into the first BiLSTM layer, as before.
    pub tcn_channels: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            input_width: crate::spline::WIDTH,
            hidden_size: 64,
            num_layers: 5,
            classes: decode::classes(),
            // Disabled by default. The TCN front-end was added as an
            // experiment, then compromised (flat dilation=1 instead of the
            // standard growing-dilation TCN recipe) to route around a real
            // CubeCL crash on dilation>1 — "can't allocate buffer of size:
            // 18446744073709549568" (~= u64::MAX, an integer-underflow
            // signature in `cubecl-wgpu`/`cubecl-runtime`, a separate repo
            // from `burn`). That compromised, non-standard front-end was
            // then baked into every training run for the rest of the night
            // with no controlled comparison against the proven
            // no-front-end architecture (armrest's own `training.py`, the
            // reference paper) — a real, previously-unexamined deviation
            // from "well-established techniques" while the model was stuck
            // in the blank-collapse basin regardless. Set back to
            // `Some(32)` (or pass `TrainConfig::tcn_channels_override`) to
            // re-enable it for comparison.
            tcn_channels: None,
        }
    }
}

/// Kernel width and dilation of each TCN front-end conv layer — dilation
/// doubling per layer (1, 2) grows the effective receptive field
/// exponentially with depth rather than linearly, the standard TCN recipe
/// (Bai et al. 2018). `Same` padding (see `Recognizer::new`) keeps sequence
/// length exactly unchanged, since CTC's feasibility check
/// (`steps >= 2*labels.len()+1`, see `train::prepare_samples`) depends on it.
const TCN_KERNEL: usize = 5;
const TCN_LAYERS: usize = 2;

/// Dropout rate applied after every BiLSTM layer during training. The
/// reference paper ("Fast Multi-language LSTM-based Online Handwriting
/// Recognition", §3.1) and armrest's own `training.py` both use 0.5, but
/// their corpus is far larger than ours (~2,580 samples) — 0.5 after every
/// one of 5 BiLSTM layers plus the TCN front-end is unusually heavy
/// stacked regularization for this little data, flagged by a pipeline
/// audit as a plausible reason CTC's already-diffuse early training signal
/// was taking unusually long to sharpen into real predictions. Lowered as
/// a genuinely different lever from tonight's LR/architecture search.
const DROPOUT_RATE: f64 = 0.2;

/// The full recognition network: `Config::num_layers` stacked bidirectional
/// LSTM layers (each followed by dropout when training), then a
/// per-timestep `Dense(classes) + softmax`.
///
/// The `Config` fields are duplicated here as plain `usize`s rather than
/// nesting a `Config` value directly: Burn's `Module` derive only treats a
/// closed list of primitive types (bool, usize, f32, ...) as
/// non-parameter "constant" fields — an arbitrary struct like `Config`
/// doesn't automatically qualify.
#[derive(Module, Debug)]
pub struct Recognizer<B: Backend> {
    tcn: Vec<Conv1d<B>>,
    layers: Vec<FusedBiLstmLayer<B>>,
    norms: Vec<BatchNorm<B>>,
    dropout: Dropout,
    dense: Linear<B>,
    input_width: usize,
    hidden_size: usize,
    num_layers: usize,
    classes: usize,
    tcn_channels: Option<usize>,
}

impl<B: FusedLstm> Recognizer<B> {
    pub fn new(config: Config, device: &B::Device) -> Self {
        let mut tcn = Vec::new();
        let mut lstm_input_dim = config.input_width;
        if let Some(channels) = config.tcn_channels {
            let mut in_ch = config.input_width;
            for _ in 0..TCN_LAYERS {
                // Flat dilation=1 for every layer, not the standard 1,2,4,...
                // TCN growth: dilation>1 hits a real CubeCL bug (integer
                // underflow in the dilated-conv autotune path, independent
                // of batch size — see `Config::default`'s old comment / the
                // session history). Two kernel-5, dilation-1 layers still
                // reach ~9 timesteps effectively (layer 2 extends layer 1's
                // already-aggregated 5-wide window by 4 more) — less
                // efficient than dilation growth, but avoids the crash
                // entirely without needing a cubecl-level fix.
                let dilation = 1;
                tcn.push(
                    Conv1dConfig::new(in_ch, channels, TCN_KERNEL)
                        .with_dilation(dilation)
                        .with_padding(PaddingConfig1d::Same)
                        .init(device),
                );
                in_ch = channels;
            }
            lstm_input_dim = channels;
        }

        let mut layers = Vec::with_capacity(config.num_layers);
        let mut norms = Vec::with_capacity(config.num_layers);
        let mut in_dim = lstm_input_dim;
        for _ in 0..config.num_layers {
            layers.push(FusedBiLstmLayer::new(in_dim, config.hidden_size, device));
            in_dim = config.hidden_size * 2;
            norms.push(BatchNormConfig::new(in_dim).init(device));
        }
        let dense = LinearConfig::new(in_dim, config.classes).init(device);
        let dropout = DropoutConfig::new(DROPOUT_RATE).init();
        Recognizer {
            tcn,
            layers,
            norms,
            dropout,
            dense,
            input_width: config.input_width,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            classes: config.classes,
            tcn_channels: config.tcn_channels,
        }
    }

    pub fn config(&self) -> Config {
        Config {
            input_width: self.input_width,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            classes: self.classes,
            tcn_channels: self.tcn_channels,
        }
    }

    /// `xs`: `[batch, steps, input_width]` -> per-step class logits
    /// `[batch, steps, classes]`, *before* the softmax. Burn's `CTCLoss`
    /// wants log-probabilities (`log_softmax`), not raw logits or softmax
    /// output — see `crate::train`.
    pub fn forward_logits(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_traced(xs).0
    }

    /// Same as [`Self::forward_logits`], plus named intermediate activations
    /// (`input`, optional `tcn0`/`tcn1`, `lstm0`..`lstmN`, `logits`) so a
    /// training-time probe can watch layer scale without a second forward.
    /// See `crate::probe` for why this exists.
    pub fn forward_traced(&self, xs: Tensor<B, 3>) -> (Tensor<B, 3>, Vec<(String, Tensor<B, 3>)>) {
        // TCN front-end: local pattern detection (corners, direction
        // reversals — the geometric primitives a stroke is built from)
        // directly from nearby timesteps, before the BiLSTM stack has to
        // do that *and* long-range sequence modeling at every layer. Conv1d
        // wants channels at dim 1 ([batch, channels, steps]); our data has
        // them last, so swap around each call, same as the BatchNorm calls
        // below.
        let mut traces = Vec::new();
        traces.push(("input".to_string(), xs.clone()));
        let mut h = xs;
        for (i, conv) in self.tcn.iter().enumerate() {
            let conv_in = h.swap_dims(1, 2);
            let conv_out = relu(conv.forward(conv_in));
            h = conv_out.swap_dims(1, 2);
            traces.push((format!("tcn{i}"), h.clone()));
        }

        for (i, (layer, norm)) in self.layers.iter().zip(&self.norms).enumerate() {
            let out = layer.forward(h);
            // BatchNorm expects the channel axis at dim 1 ([batch, channels,
            // ...]); our LSTM output has it last ([batch, steps, hidden*2]),
            // so swap steps/channels around the call.
            let out = norm.forward(out.swap_dims(1, 2)).swap_dims(1, 2);
            h = self.dropout.forward(out);
            traces.push((format!("lstm{i}"), h.clone()));
        }
        let logits = self.dense.forward(h);
        traces.push(("logits".to_string(), logits.clone()));
        (logits, traces)
    }

    /// `xs`: `[batch, steps, input_width]` -> per-step class probabilities
    /// `[batch, steps, classes]` (already softmaxed). For inference.
    pub fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        softmax(self.forward_logits(xs), 2)
    }
}
