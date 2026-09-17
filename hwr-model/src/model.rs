//! HAT (Lodh et al. 2025, "A Transformer Based Handwriting Recognition
//! System Jointly Using Online and Offline Features"): early fusion of a
//! rasterized glyph and the pen trajectory in a shared `d`-dimensional
//! token space.
//!
//! Pipeline (paper Fig. 2 / Eqs. 1–9):
//! 1. **Image patch encoder** — 224×224 crop → 7×7 tokens → project to `d`.
//!    The paper uses pretrained Swin-B (last stage `7×7×1024`). Burn 0.21
//!    has no ImageNet Swin weights, and 88M frozen/trainable params would
//!    not fit next to PBT on this machine, so the same token geometry is
//!    produced by a stride-2 conv stem (five layers, 224→7) ending at 1024
//!    channels, then `W_p ∈ R^{1024×d}` as in Eq. (1).
//! 2. **Latent cross-attention** — `L=64` learnable latents attend to the
//!    patch tokens (Perceiver-IO style, Eqs. 2a–2b).
//! 3. **Stroke encoder** — `(x, y)` concatenated with a pen-state embedding
//!    of size `d/8`, linear + BatchNorm + Dropout, rotary positional
//!    encoding, `N_stk`-layer Transformer, residual 2-layer MLP (Eqs. 3–6).
//! 4. **Cross-modal query** — stroke tokens attend to image latents
//!    (Eqs. 7a–7b).
//! 5. **Head** — the paper pools with attention (Eq. 8) and classifies
//!    isolated characters with cross-entropy (Eqs. 9–10). We still need
//!    *unsegmented words*, which that paper lists as future work, so the
//!    same `W_c` is applied **per fused stroke token** and trained with
//!    CTC (the previous recognizer's loss). [`Recognizer::forward_isolated`]
//!    keeps the paper's pooled classifier for single-glyph use.
//!
//! LSTM/TCN checkpoints are a different `Module` layout and will not load.

use burn::module::{Module, Param};
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::transformer::{
    PositionWiseFeedForward, PositionWiseFeedForwardConfig, TransformerEncoder,
    TransformerEncoderConfig, TransformerEncoderInput, TransformerEncoderLayer,
};
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer,
    LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d, RotaryEncoding,
    RotaryEncodingConfig,
};
use burn::tensor::activation::{relu, softmax, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor, TensorData};

use crate::decode;
use crate::spline;

/// Device tensors for one HAT forward from [`spline::pack_hat_batch`].
pub fn packed_inputs<B: Backend>(
    strokes: Vec<f32>,
    images: Vec<f32>,
    pad_mask: Vec<f32>,
    launch: usize,
    max_steps: usize,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 2, Bool>) {
    let strokes = Tensor::<B, 3>::from_data(
        TensorData::new(strokes, [launch, max_steps, spline::STROKE_DIM]),
        device,
    );
    let images = Tensor::<B, 4>::from_data(
        TensorData::new(
            images,
            [
                launch,
                spline::IMAGE_CHANNELS,
                spline::IMAGE_SIZE,
                spline::IMAGE_SIZE,
            ],
        ),
        device,
    );
    let pad_mask = Tensor::<B, 2>::from_data(TensorData::new(pad_mask, [launch, max_steps]), device)
        .greater_elem(0.5);
    (strokes, images, pad_mask)
}

/// Paper §3: `d = 256`, `L = 64` latents, dropout 0.1 on the stroke path.
/// `N_stk = 4` matches the reported ~4.3M stroke-only parameter count.
const D_MODEL: usize = 256;
const N_HEADS: usize = 8;
const D_FF: usize = 1024;
const N_STROKE_LAYERS: usize = 4;
const N_LATENT_LAYERS: usize = 2;
const N_LATENTS: usize = 64;
const DROPOUT: f64 = 0.1;
const MAX_SEQ_LEN: usize = 512;
const PATCH_CHANNELS: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_stroke_layers: usize,
    pub n_latent_layers: usize,
    pub n_latents: usize,
    pub classes: usize,
    pub dropout: f64,
    pub max_seq_len: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            d_model: D_MODEL,
            n_heads: N_HEADS,
            d_ff: D_FF,
            n_stroke_layers: N_STROKE_LAYERS,
            n_latent_layers: N_LATENT_LAYERS,
            n_latents: N_LATENTS,
            classes: decode::classes(),
            dropout: DROPOUT,
            max_seq_len: MAX_SEQ_LEN,
        }
    }
}

#[derive(Module, Debug)]
pub struct Recognizer<B: Backend> {
    img_convs: Vec<Conv2d<B>>,
    img_proj: Linear<B>,
    latents: Param<Tensor<B, 2>>,
    latent_mha: Vec<MultiHeadAttention<B>>,
    latent_tf: Vec<TransformerEncoderLayer<B>>,
    latent_norm: LayerNorm<B>,
    pen_embed: Embedding<B>,
    stroke_proj: Linear<B>,
    stroke_bn: BatchNorm<B>,
    dropout: Dropout,
    rope: RotaryEncoding<B>,
    stroke_encoder: TransformerEncoder<B>,
    stroke_ff_norm: LayerNorm<B>,
    stroke_ff: PositionWiseFeedForward<B>,
    cross_mha: MultiHeadAttention<B>,
    cross_tf: TransformerEncoderLayer<B>,
    pool_w1: Linear<B>,
    pool_w2: Linear<B>,
    classifier: Linear<B>,
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    n_stroke_layers: usize,
    n_latent_layers: usize,
    n_latents: usize,
    classes: usize,
    max_seq_len: usize,
}

impl<B: Backend> Recognizer<B> {
    pub fn new(config: Config, device: &B::Device) -> Self {
        let d = config.d_model;
        let tf_cfg = TransformerEncoderConfig::new(d, config.d_ff, config.n_heads, 1)
            .with_dropout(config.dropout);
        let mha_cfg = MultiHeadAttentionConfig::new(d, config.n_heads).with_dropout(config.dropout);

        let mut img_convs = Vec::with_capacity(5);
        let mut cin = spline::IMAGE_CHANNELS;
        for &cout in &[64usize, 128, 256, 512, PATCH_CHANNELS] {
            img_convs.push(
                Conv2dConfig::new([cin, cout], [3, 3])
                    .with_stride([2, 2])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
            );
            cin = cout;
        }

        let mut latent_mha = Vec::with_capacity(config.n_latent_layers);
        let mut latent_tf = Vec::with_capacity(config.n_latent_layers);
        for _ in 0..config.n_latent_layers {
            latent_mha.push(mha_cfg.init(device));
            latent_tf.push(TransformerEncoderLayer::new(&tf_cfg, device));
        }

        let pen_dim = d / 8;
        Recognizer {
            img_convs,
            img_proj: LinearConfig::new(PATCH_CHANNELS, d).init(device),
            latents: Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init([config.n_latents, d], device),
            latent_mha,
            latent_tf,
            latent_norm: LayerNormConfig::new(d).init(device),
            pen_embed: EmbeddingConfig::new(2, pen_dim).init(device),
            stroke_proj: LinearConfig::new(2 + pen_dim, d).init(device),
            stroke_bn: BatchNormConfig::new(d).init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
            rope: RotaryEncodingConfig::new(config.max_seq_len, d).init(device),
            stroke_encoder: TransformerEncoderConfig::new(
                d,
                config.d_ff,
                config.n_heads,
                config.n_stroke_layers,
            )
            .with_dropout(config.dropout)
            .init(device),
            stroke_ff_norm: LayerNormConfig::new(d).init(device),
            stroke_ff: PositionWiseFeedForwardConfig::new(d, config.d_ff)
                .with_dropout(config.dropout)
                .init(device),
            cross_mha: mha_cfg.init(device),
            cross_tf: TransformerEncoderLayer::new(&tf_cfg, device),
            pool_w1: LinearConfig::new(d, d).init(device),
            pool_w2: LinearConfig::new(d, 1).init(device),
            classifier: LinearConfig::new(d, config.classes).init(device),
            d_model: d,
            n_heads: config.n_heads,
            d_ff: config.d_ff,
            n_stroke_layers: config.n_stroke_layers,
            n_latent_layers: config.n_latent_layers,
            n_latents: config.n_latents,
            classes: config.classes,
            max_seq_len: config.max_seq_len,
        }
    }

    pub fn config(&self) -> Config {
        Config {
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_ff: self.d_ff,
            n_stroke_layers: self.n_stroke_layers,
            n_latent_layers: self.n_latent_layers,
            n_latents: self.n_latents,
            classes: self.classes,
            dropout: DROPOUT,
            max_seq_len: self.max_seq_len,
        }
    }

    /// `strokes`: `[batch, steps, 3]` `(x, y, pen_up)`;
    /// `images`: `[batch, 3, 224, 224]`;
    /// `pad_mask`: `[batch, steps]`, `true` = padded key (Burn MHA convention).
    /// Returns per-step class logits `[batch, steps, classes]` (CTC).
    pub fn forward_logits(
        &self,
        strokes: Tensor<B, 3>,
        images: Tensor<B, 4>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        self.forward_traced(strokes, images, pad_mask).0
    }

    pub fn forward_traced(
        &self,
        strokes: Tensor<B, 3>,
        images: Tensor<B, 4>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> (Tensor<B, 3>, Vec<(String, Tensor<B, 3>)>) {
        let mut traces = Vec::new();
        traces.push(("input".to_string(), strokes.clone()));

        let z = self.encode_image(images);
        traces.push(("latents".to_string(), z.clone()));

        let estroke = self.encode_strokes(strokes, pad_mask.clone());
        traces.push(("strokes".to_string(), estroke.clone()));

        let fused = self.cross_modal(estroke, z, pad_mask);
        traces.push(("fused".to_string(), fused.clone()));

        let logits = self.classifier.forward(fused);
        traces.push(("logits".to_string(), logits.clone()));
        (logits, traces)
    }

    /// Paper Eqs. 8–9: attention-pool the fused stroke tokens, then the
    /// linear classifier. Isolated-character logits `[batch, classes]`.
    pub fn forward_isolated(
        &self,
        strokes: Tensor<B, 3>,
        images: Tensor<B, 4>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        let tokens = self.fused_tokens(strokes, images, pad_mask.clone());
        let pooled = self.attention_pool(tokens, pad_mask);
        self.classifier.forward(pooled)
    }

    /// Per-step softmax, for greedy / beam decode.
    pub fn forward(
        &self,
        strokes: Tensor<B, 3>,
        images: Tensor<B, 4>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        softmax(self.forward_logits(strokes, images, pad_mask), 2)
    }

    fn fused_tokens(
        &self,
        strokes: Tensor<B, 3>,
        images: Tensor<B, 4>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let z = self.encode_image(images);
        let estroke = self.encode_strokes(strokes, pad_mask.clone());
        self.cross_modal(estroke, z, pad_mask)
    }

    /// Eq. (1) plus the conv stem that stands in for Swin-B's last stage.
    fn encode_image(&self, images: Tensor<B, 4>) -> Tensor<B, 3> {
        let mut h = images;
        for conv in &self.img_convs {
            h = relu(conv.forward(h));
        }
        let [b, c, height, width] = h.dims();
        let patches = h.reshape([b, c, height * width]).swap_dims(1, 2);
        let ep = self.img_proj.forward(patches);

        let [batch, _, _] = ep.dims();
        let mut z = self
            .latents
            .val()
            .unsqueeze::<3>()
            .repeat_dim(0, batch);
        for (mha, tf) in self.latent_mha.iter().zip(&self.latent_tf) {
            let z_tilde = mha.forward(MhaInput::new(z.clone(), ep.clone(), ep.clone())).context;
            z = tf.forward(z.add(z_tilde), None, None);
        }
        self.latent_norm.forward(z)
    }

    /// Eqs. (3)–(6).
    fn encode_strokes(
        &self,
        strokes: Tensor<B, 3>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [b, t, _] = strokes.dims();
        let xy = strokes.clone().slice([0..b, 0..t, 0..2]);
        let pen = strokes
            .slice([0..b, 0..t, 2..3])
            .reshape([b, t])
            .clamp(0.0, 1.0)
            .round()
            .int();
        let pen_e = self.pen_embed.forward(pen);
        let st = Tensor::cat(vec![xy, pen_e], 2);
        let mut es = self.stroke_proj.forward(st);
        es = self.stroke_bn.forward(es.swap_dims(1, 2)).swap_dims(1, 2);
        es = self.dropout.forward(es);
        es = self.rope.forward(es);

        let mut enc_in = TransformerEncoderInput::new(es);
        if let Some(mask) = pad_mask.clone() {
            enc_in = enc_in.mask_pad(mask);
        }
        let h = self.stroke_encoder.forward(enc_in);
        h.clone().add(self.stroke_ff.forward(self.stroke_ff_norm.forward(h)))
    }

    /// Eqs. (7a)–(7b).
    fn cross_modal(
        &self,
        estroke: Tensor<B, 3>,
        z: Tensor<B, 3>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let tilde = self
            .cross_mha
            .forward(MhaInput::new(estroke.clone(), z.clone(), z))
            .context;
        self.cross_tf
            .forward(estroke.add(tilde), pad_mask, None)
    }

    /// Eq. (8). `pad_mask` true = ignore that stroke token.
    fn attention_pool(
        &self,
        tokens: Tensor<B, 3>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 2> {
        let mut scores = self.pool_w2.forward(tanh(self.pool_w1.forward(tokens.clone())));
        if let Some(mask) = pad_mask {
            let [b, n] = mask.dims();
            let fill = mask.reshape([b, n, 1]);
            scores = scores.mask_fill(fill, -1.0e4);
        }
        let alpha = softmax(scores, 1);
        alpha.mul(tokens).sum_dim(1).squeeze_dim::<2>(1)
    }
}
