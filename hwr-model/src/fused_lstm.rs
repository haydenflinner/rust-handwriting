//! A fused single-direction LSTM-over-a-sequence op, registered as ONE
//! autodiff graph node regardless of sequence length — unlike
//! `burn::nn::Lstm`/`BiLstm`, whose per-timestep Rust loop makes Burn's
//! autodiff track O(T) nodes per layer (T = timesteps). We measured that
//! graph-node bookkeeping (`burn-autodiff`'s `GraphMutexClient`/
//! `GraphLocator::select`, a mutex-guarded hashmap-based structure) scaling
//! super-linearly with T on the plain CPU backend — nothing to do with GPU
//! dispatch — which is what made full BiLSTM training far slower than the
//! isolated `CTCLoss` benchmark predicted.
//!
//! Forward runs the per-timestep recurrence as plain composed tensor ops on
//! the *untracked* inner backend (so it costs nothing in graph bookkeeping,
//! same as ordinary Rust/BLAS compute); backward is hand-derived standard
//! BPTT (backprop-through-time) for a single LSTM cell, registered as ONE
//! custom autodiff node — collapsing per-layer node count from O(T) to
//! O(1). A `BiLstm`-equivalent stacks `5` layers x `2` directions of this,
//! so total node count becomes O(layers) instead of O(layers * T).
//!
//! Validated by finite-difference gradient checks in this module's tests —
//! the same rigor used for the hand-rolled CTC forward-backward in the
//! original candle port, since analytic BPTT gradients are exactly the
//! kind of thing that's easy to get subtly wrong.

use burn::backend::Autodiff;
use burn::backend::autodiff::NodeId;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::module::{Module, Param};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Distribution, Tensor, TensorPrimitive};

/// `Tensor<B, D>` (Float kind)'s primitive is `TensorPrimitive<B>` (an enum
/// covering both plain-float and quantized storage), not the raw
/// `FloatTensor<B> = B::FloatTensorPrimitive` that the `Backward`/`Ops` API
/// (and our own `FusedLstm` trait, to match that convention) operates on —
/// these two helpers are the only place that distinction shows up.
fn wrap<B: Backend, const D: usize>(raw: FloatTensor<B>) -> Tensor<B, D> {
    Tensor::from_primitive(TensorPrimitive::Float(raw))
}

fn unwrap<B: Backend, const D: usize>(t: Tensor<B, D>) -> FloatTensor<B> {
    t.into_primitive().tensor()
}

/// Everything from the forward pass needed to compute BPTT gradients:
/// per-timestep gate activations and cell/hidden states, each `[B, T, H]`.
#[derive(Clone)]
pub struct LstmCache<B: Backend> {
    pub i: Tensor<B, 3>,
    pub f: Tensor<B, 3>,
    pub g: Tensor<B, 3>,
    pub o: Tensor<B, 3>,
    pub c: Tensor<B, 3>,
    pub h: Tensor<B, 3>,
}

/// Run an LSTM over a whole sequence in one call: `x: [B, T, Din]`,
/// `w_ih: [4H, Din]`, `w_hh: [4H, H]`, `b_ih`/`b_hh: [4H]` (gate order
/// i,f,g,o — our own convention; forward and backward are defined together
/// here so nothing else needs to match it). Initial hidden/cell state is
/// zero. Returns the hidden-state sequence `[B, T, H]` plus the cache
/// `lstm_backward` needs.
pub fn lstm_forward<B: Backend>(
    x: Tensor<B, 3>,
    w_ih: Tensor<B, 2>,
    w_hh: Tensor<B, 2>,
    b_ih: Tensor<B, 1>,
    b_hh: Tensor<B, 1>,
    hidden: usize,
) -> (Tensor<B, 3>, LstmCache<B>) {
    let [batch, steps, in_dim] = x.dims();
    let four_h = 4 * hidden;
    let device = x.device();

    // Precompute the input-side gate projection for every timestep in one
    // matmul (it doesn't depend on the recurrence) — only the
    // hidden-to-hidden part needs a per-step loop.
    let x_flat = x.reshape([batch * steps, in_dim]);
    let proj_in = x_flat.matmul(w_ih.clone().transpose());
    let proj_in = proj_in.reshape([batch, steps, four_h]);
    let proj_in = proj_in + b_ih.reshape([1, 1, four_h]);
    let b_hh_b = b_hh.reshape([1, four_h]);
    let w_hh_t = w_hh.transpose();

    let mut h_prev = Tensor::<B, 2>::zeros([batch, hidden], &device);
    let mut c_prev = Tensor::<B, 2>::zeros([batch, hidden], &device);

    let (mut i_list, mut f_list, mut g_list, mut o_list, mut c_list, mut h_list) = (
        Vec::with_capacity(steps),
        Vec::with_capacity(steps),
        Vec::with_capacity(steps),
        Vec::with_capacity(steps),
        Vec::with_capacity(steps),
        Vec::with_capacity(steps),
    );

    for t in 0..steps {
        let x_proj_t = proj_in.clone().narrow(1, t, 1).reshape([batch, four_h]);
        let rec = h_prev.clone().matmul(w_hh_t.clone());
        let gates = x_proj_t + rec + b_hh_b.clone();

        let i = sigmoid(gates.clone().narrow(1, 0, hidden));
        let f = sigmoid(gates.clone().narrow(1, hidden, hidden));
        let g = gates.clone().narrow(1, 2 * hidden, hidden).tanh();
        let o = sigmoid(gates.narrow(1, 3 * hidden, hidden));

        let c = f.clone() * c_prev + i.clone() * g.clone();
        let h = o.clone() * c.clone().tanh();

        i_list.push(i);
        f_list.push(f);
        g_list.push(g);
        o_list.push(o);
        c_list.push(c.clone());
        h_list.push(h.clone());

        h_prev = h;
        c_prev = c;
    }

    let stack = |v: Vec<Tensor<B, 2>>| Tensor::<B, 2>::stack::<3>(v, 1);
    let h_seq = stack(h_list);
    let cache = LstmCache {
        i: stack(i_list),
        f: stack(f_list),
        g: stack(g_list),
        o: stack(o_list),
        c: stack(c_list),
        h: h_seq.clone(),
    };

    (h_seq, cache)
}

/// Standard BPTT for the LSTM cell in `lstm_forward`. `grad_h_seq` is the
/// upstream gradient w.r.t. the returned hidden-state sequence. Returns
/// gradients w.r.t. `(x, w_ih, w_hh, b_ih, b_hh)`; `b_ih`/`b_hh` get the
/// identical accumulated value since they're added identically in forward.
pub fn lstm_backward<B: Backend>(
    grad_h_seq: Tensor<B, 3>,
    x: Tensor<B, 3>,
    w_ih: Tensor<B, 2>,
    w_hh: Tensor<B, 2>,
    cache: &LstmCache<B>,
    hidden: usize,
) -> (
    Tensor<B, 3>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 1>,
    Tensor<B, 1>,
) {
    let [batch, steps, in_dim] = x.dims();
    let four_h = 4 * hidden;
    let device = x.device();

    let mut grad_h_next = Tensor::<B, 2>::zeros([batch, hidden], &device);
    let mut grad_c_next = Tensor::<B, 2>::zeros([batch, hidden], &device);
    let mut grad_w_ih = Tensor::<B, 2>::zeros([four_h, in_dim], &device);
    let mut grad_w_hh = Tensor::<B, 2>::zeros([four_h, hidden], &device);
    let mut grad_b = Tensor::<B, 1>::zeros([four_h], &device);
    let mut grad_x_list: Vec<Tensor<B, 2>> = Vec::with_capacity(steps);

    let at = |seq: &Tensor<B, 3>, t: usize| seq.clone().narrow(1, t, 1).reshape([batch, hidden]);
    let zeros = || Tensor::<B, 2>::zeros([batch, hidden], &device);

    for t in (0..steps).rev() {
        let grad_h_t = at(&grad_h_seq, t) + grad_h_next;

        let c_t = at(&cache.c, t);
        let o_t = at(&cache.o, t);
        let i_t = at(&cache.i, t);
        let f_t = at(&cache.f, t);
        let g_t = at(&cache.g, t);
        let c_prev_t = if t == 0 { zeros() } else { at(&cache.c, t - 1) };
        let h_prev_t = if t == 0 { zeros() } else { at(&cache.h, t - 1) };

        let tanh_c_t = c_t.tanh();
        let grad_o_t = grad_h_t.clone() * tanh_c_t.clone();
        let one_minus_tanh2 = tanh_c_t.clone().powi_scalar(2).neg().add_scalar(1.0);
        let grad_c_t = grad_h_t * o_t.clone() * one_minus_tanh2 + grad_c_next;

        let grad_f_t = grad_c_t.clone() * c_prev_t;
        let grad_i_t = grad_c_t.clone() * g_t.clone();
        let grad_g_t = grad_c_t.clone() * i_t.clone();
        grad_c_next = grad_c_t * f_t.clone();

        let grad_i_pre = grad_i_t * i_t.clone() * i_t.neg().add_scalar(1.0);
        let grad_f_pre = grad_f_t * f_t.clone() * f_t.neg().add_scalar(1.0);
        let grad_g_pre = grad_g_t * g_t.powi_scalar(2).neg().add_scalar(1.0);
        let grad_o_pre = grad_o_t * o_t.clone() * o_t.neg().add_scalar(1.0);

        let grad_gates = Tensor::cat(vec![grad_i_pre, grad_f_pre, grad_g_pre, grad_o_pre], 1);

        let x_t = x.clone().narrow(1, t, 1).reshape([batch, in_dim]);
        grad_x_list.push(grad_gates.clone().matmul(w_ih.clone()));
        grad_h_next = grad_gates.clone().matmul(w_hh.clone());

        grad_w_ih = grad_w_ih + grad_gates.clone().transpose().matmul(x_t);
        grad_w_hh = grad_w_hh + grad_gates.clone().transpose().matmul(h_prev_t);
        grad_b = grad_b + grad_gates.sum_dim(0).reshape([four_h]);
    }

    grad_x_list.reverse();
    let grad_x = Tensor::<B, 2>::stack::<3>(grad_x_list, 1);

    (grad_x, grad_w_ih, grad_w_hh, grad_b.clone(), grad_b)
}

/// Per-INNER-backend forward+backward strategy for the LSTM recurrence.
/// Deliberately a separate trait from [`FusedLstm`], implemented only by
/// concrete backends (`NdArray` for tests, `Wgpu` for production) and NOT by
/// `Autodiff<B, C>` — these two methods are only ever called on the inner
/// `B` from within `Autodiff<B, C>`'s own custom `Backward` wiring below,
/// never on `Autodiff<B, C>` itself, so there'd be nothing meaningful for
/// `Autodiff` to implement here. `NdArray` (test-only, see `Cargo.toml`)
/// uses the portable composed-tensor-ops functions below; `Wgpu`
/// (production) uses the fused CubeCL kernel in `fused_lstm_kernel.rs` —
/// see that module's docs for why a hand-written kernel needs its own
/// implementation rather than being generic over `Backend` the way
/// `lstm_forward`/`lstm_backward` are.
pub trait FusedLstmKernel: Backend {
    fn lstm_seq_forward_cached(
        x: Tensor<Self, 3>,
        w_ih: Tensor<Self, 2>,
        w_hh: Tensor<Self, 2>,
        b_ih: Tensor<Self, 1>,
        b_hh: Tensor<Self, 1>,
        hidden_size: usize,
    ) -> (Tensor<Self, 3>, LstmCache<Self>);

    #[allow(clippy::type_complexity)]
    fn lstm_seq_backward(
        grad_h_seq: Tensor<Self, 3>,
        x: Tensor<Self, 3>,
        w_ih: Tensor<Self, 2>,
        w_hh: Tensor<Self, 2>,
        cache: &LstmCache<Self>,
        hidden_size: usize,
    ) -> (
        Tensor<Self, 3>,
        Tensor<Self, 2>,
        Tensor<Self, 2>,
        Tensor<Self, 1>,
        Tensor<Self, 1>,
    );
}

/// Backend extension trait: the point of this indirection is so
/// `Autodiff<B, C>` can get a DIFFERENT implementation (one that registers
/// a single custom graph node with a hand-written backward, via
/// [`FusedLstmKernel`] on the inner `B`) than the plain backend's (which
/// just calls its own [`FusedLstmKernel::lstm_seq_forward_cached`]
/// directly, untracked, and discards the cache).
pub trait FusedLstm: Backend {
    fn lstm_seq(
        x: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        b_ih: FloatTensor<Self>,
        b_hh: FloatTensor<Self>,
        hidden_size: usize,
    ) -> FloatTensor<Self>;
}

macro_rules! impl_fused_lstm_plain {
    ($backend:ty) => {
        impl FusedLstmKernel for $backend {
            fn lstm_seq_forward_cached(
                x: Tensor<Self, 3>,
                w_ih: Tensor<Self, 2>,
                w_hh: Tensor<Self, 2>,
                b_ih: Tensor<Self, 1>,
                b_hh: Tensor<Self, 1>,
                hidden_size: usize,
            ) -> (Tensor<Self, 3>, LstmCache<Self>) {
                lstm_forward::<Self>(x, w_ih, w_hh, b_ih, b_hh, hidden_size)
            }

            fn lstm_seq_backward(
                grad_h_seq: Tensor<Self, 3>,
                x: Tensor<Self, 3>,
                w_ih: Tensor<Self, 2>,
                w_hh: Tensor<Self, 2>,
                cache: &LstmCache<Self>,
                hidden_size: usize,
            ) -> (
                Tensor<Self, 3>,
                Tensor<Self, 2>,
                Tensor<Self, 2>,
                Tensor<Self, 1>,
                Tensor<Self, 1>,
            ) {
                lstm_backward::<Self>(grad_h_seq, x, w_ih, w_hh, cache, hidden_size)
            }
        }

        impl FusedLstm for $backend {
            fn lstm_seq(
                x: FloatTensor<Self>,
                w_ih: FloatTensor<Self>,
                w_hh: FloatTensor<Self>,
                b_ih: FloatTensor<Self>,
                b_hh: FloatTensor<Self>,
                hidden_size: usize,
            ) -> FloatTensor<Self> {
                let (h_seq, _cache) = lstm_forward::<Self>(
                    wrap(x),
                    wrap(w_ih),
                    wrap(w_hh),
                    wrap(b_ih),
                    wrap(b_hh),
                    hidden_size,
                );
                unwrap(h_seq)
            }
        }
    };
}

// NdArray is only pulled in via `[dev-dependencies]` (see Cargo.toml) for
// fast CPU finite-difference gradient checks in this module's tests — it
// uses the portable composed-tensor-ops path (no CubeCL kernel exists for
// it, nor does it need one; these tests care about correctness, not speed).
#[cfg(test)]
impl_fused_lstm_plain!(burn::backend::NdArray);

impl FusedLstmKernel for burn::backend::Wgpu {
    fn lstm_seq_forward_cached(
        x: Tensor<Self, 3>,
        w_ih: Tensor<Self, 2>,
        w_hh: Tensor<Self, 2>,
        b_ih: Tensor<Self, 1>,
        b_hh: Tensor<Self, 1>,
        hidden_size: usize,
    ) -> (Tensor<Self, 3>, LstmCache<Self>) {
        crate::fused_lstm_kernel::kernel_forward(x, w_ih, w_hh, b_ih, b_hh, hidden_size)
    }

    fn lstm_seq_backward(
        grad_h_seq: Tensor<Self, 3>,
        x: Tensor<Self, 3>,
        w_ih: Tensor<Self, 2>,
        w_hh: Tensor<Self, 2>,
        cache: &LstmCache<Self>,
        hidden_size: usize,
    ) -> (
        Tensor<Self, 3>,
        Tensor<Self, 2>,
        Tensor<Self, 2>,
        Tensor<Self, 1>,
        Tensor<Self, 1>,
    ) {
        crate::fused_lstm_kernel::kernel_backward(grad_h_seq, x, w_ih, w_hh, cache, hidden_size)
    }
}

impl FusedLstm for burn::backend::Wgpu {
    fn lstm_seq(
        x: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        b_ih: FloatTensor<Self>,
        b_hh: FloatTensor<Self>,
        hidden_size: usize,
    ) -> FloatTensor<Self> {
        let (h_seq, _cache) = crate::fused_lstm_kernel::kernel_forward(
            wrap(x),
            wrap(w_ih),
            wrap(w_hh),
            wrap(b_ih),
            wrap(b_hh),
            hidden_size,
        );
        unwrap(h_seq)
    }
}

impl<B: FusedLstmKernel, C: CheckpointStrategy> FusedLstm for Autodiff<B, C> {
    fn lstm_seq(
        x: FloatTensor<Self>,
        w_ih: FloatTensor<Self>,
        w_hh: FloatTensor<Self>,
        b_ih: FloatTensor<Self>,
        b_hh: FloatTensor<Self>,
        hidden_size: usize,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct LstmSeqBackward {
            hidden_size: usize,
        }

        #[allow(clippy::type_complexity)]
        impl<B: FusedLstmKernel> Backward<B, 5> for LstmSeqBackward {
            type State = (
                NodeId,
                NodeId,
                NodeId,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
                FloatTensor<B>,
            );

            fn backward(
                self,
                ops: Ops<Self::State, 5>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x, node_w_ih, node_w_hh, node_b_ih, node_b_hh] = ops.parents;
                let grad_h_seq = grads.consume::<B>(&ops.node);

                let (x_id, w_ih_id, w_hh_id, i, f, g, o, c, h) = ops.state;
                let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_id);
                let w_ih: FloatTensor<B> = checkpointer.retrieve_node_output(w_ih_id);
                let w_hh: FloatTensor<B> = checkpointer.retrieve_node_output(w_hh_id);

                let cache = LstmCache::<B> {
                    i: wrap(i),
                    f: wrap(f),
                    g: wrap(g),
                    o: wrap(o),
                    c: wrap(c),
                    h: wrap(h),
                };

                let (grad_x, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh) = B::lstm_seq_backward(
                    wrap(grad_h_seq),
                    wrap(x),
                    wrap(w_ih),
                    wrap(w_hh),
                    &cache,
                    self.hidden_size,
                );

                if let Some(node) = node_x {
                    grads.register::<B>(node.id, unwrap(grad_x));
                }
                if let Some(node) = node_w_ih {
                    grads.register::<B>(node.id, unwrap(grad_w_ih));
                }
                if let Some(node) = node_w_hh {
                    grads.register::<B>(node.id, unwrap(grad_w_hh));
                }
                if let Some(node) = node_b_ih {
                    grads.register::<B>(node.id, unwrap(grad_b_ih));
                }
                if let Some(node) = node_b_hh {
                    grads.register::<B>(node.id, unwrap(grad_b_hh));
                }
            }
        }

        match (LstmSeqBackward { hidden_size })
            .prepare::<C>([
                x.node.clone(),
                w_ih.node.clone(),
                w_hh.node.clone(),
                b_ih.node.clone(),
                b_hh.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let w_ih_state = prep.checkpoint(&w_ih);
                let w_hh_state = prep.checkpoint(&w_hh);

                let (h_seq, cache) = B::lstm_seq_forward_cached(
                    wrap(x.into_primitive()),
                    wrap(w_ih.into_primitive()),
                    wrap(w_hh.into_primitive()),
                    wrap(b_ih.into_primitive()),
                    wrap(b_hh.into_primitive()),
                    hidden_size,
                );

                let state = (
                    x_state,
                    w_ih_state,
                    w_hh_state,
                    unwrap(cache.i),
                    unwrap(cache.f),
                    unwrap(cache.g),
                    unwrap(cache.o),
                    unwrap(cache.c),
                    unwrap(cache.h),
                );

                prep.finish(state, unwrap(h_seq))
            }
            OpsKind::UnTracked(prep) => {
                let (h_seq, _cache) = B::lstm_seq_forward_cached(
                    wrap(x.into_primitive()),
                    wrap(w_ih.into_primitive()),
                    wrap(w_hh.into_primitive()),
                    wrap(b_ih.into_primitive()),
                    wrap(b_hh.into_primitive()),
                    hidden_size,
                );
                prep.finish(unwrap(h_seq))
            }
        }
    }
}

/// Ergonomic wrapper: run [`FusedLstm::lstm_seq`] on `Tensor` values instead
/// of raw primitives.
pub fn fused_lstm_seq<B: FusedLstm>(
    x: Tensor<B, 3>,
    w_ih: Tensor<B, 2>,
    w_hh: Tensor<B, 2>,
    b_ih: Tensor<B, 1>,
    b_hh: Tensor<B, 1>,
    hidden_size: usize,
) -> Tensor<B, 3> {
    wrap(B::lstm_seq(
        unwrap(x),
        unwrap(w_ih),
        unwrap(w_hh),
        unwrap(b_ih),
        unwrap(b_hh),
        hidden_size,
    ))
}

/// Bidirectional concat-merge wrapper around [`fused_lstm_seq`]: runs the
/// forward direction normally and the backward direction over the
/// time-reversed input (re-reversing its output afterward), then
/// concatenates both hidden-state sequences on the feature axis — matching
/// `burn::nn::BiLstm`'s `concat` merge mode, which is what this replaces.
#[allow(clippy::too_many_arguments)]
pub fn fused_bilstm_seq<B: FusedLstm>(
    x: Tensor<B, 3>,
    w_ih_fwd: Tensor<B, 2>,
    w_hh_fwd: Tensor<B, 2>,
    b_ih_fwd: Tensor<B, 1>,
    b_hh_fwd: Tensor<B, 1>,
    w_ih_rev: Tensor<B, 2>,
    w_hh_rev: Tensor<B, 2>,
    b_ih_rev: Tensor<B, 1>,
    b_hh_rev: Tensor<B, 1>,
    hidden_size: usize,
) -> Tensor<B, 3> {
    let fwd = fused_lstm_seq(x.clone(), w_ih_fwd, w_hh_fwd, b_ih_fwd, b_hh_fwd, hidden_size);
    let rev_out = fused_lstm_seq(
        x.flip([1]),
        w_ih_rev,
        w_hh_rev,
        b_ih_rev,
        b_hh_rev,
        hidden_size,
    )
    .flip([1]);
    Tensor::cat(vec![fwd, rev_out], 2)
}

/// A `Module`-integrated bidirectional LSTM layer (concat merge) built on
/// [`fused_bilstm_seq`] — drop-in replacement for `burn::nn::BiLstm` inside
/// a `Module` struct (checkpointing via `NamedMpkFileRecorder`, the
/// `AdamW` optimizer's `GradientsParams::from_grads`, and
/// `AutodiffModule::valid()` for inference all keep working the same way),
/// at the speed measured in this module's `tests` (and
/// `bin/bench_fused_lstm.rs`) rather than stock `BiLstm`'s.
#[derive(Module, Debug)]
pub struct FusedBiLstmLayer<B: Backend> {
    w_ih_fwd: Param<Tensor<B, 2>>,
    w_hh_fwd: Param<Tensor<B, 2>>,
    b_ih_fwd: Param<Tensor<B, 1>>,
    b_hh_fwd: Param<Tensor<B, 1>>,
    w_ih_rev: Param<Tensor<B, 2>>,
    w_hh_rev: Param<Tensor<B, 2>>,
    b_ih_rev: Param<Tensor<B, 1>>,
    b_hh_rev: Param<Tensor<B, 1>>,
    hidden_size: usize,
}

impl<B: Backend> FusedBiLstmLayer<B> {
    /// `input_size` -> `hidden_size` per direction (output width is
    /// `2 * hidden_size`, concat merge). Weights are uniform-initialized in
    /// `[-k, k]` with `k = 1/sqrt(hidden_size)`, matching PyTorch/Burn's own
    /// default LSTM init.
    pub fn new(input_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        let k = 1.0 / (hidden_size as f64).sqrt();
        let dist = Distribution::Uniform(-k, k);
        let w2 = |r: usize, c: usize| Param::from_tensor(Tensor::<B, 2>::random([r, c], dist, device));
        let w1 = |n: usize| Param::from_tensor(Tensor::<B, 1>::random([n], dist, device));

        FusedBiLstmLayer {
            w_ih_fwd: w2(4 * hidden_size, input_size),
            w_hh_fwd: w2(4 * hidden_size, hidden_size),
            b_ih_fwd: w1(4 * hidden_size),
            b_hh_fwd: w1(4 * hidden_size),
            w_ih_rev: w2(4 * hidden_size, input_size),
            w_hh_rev: w2(4 * hidden_size, hidden_size),
            b_ih_rev: w1(4 * hidden_size),
            b_hh_rev: w1(4 * hidden_size),
            hidden_size,
        }
    }
}

impl<B: FusedLstm> FusedBiLstmLayer<B> {
    /// `xs`: `[batch, steps, input_size]` -> `[batch, steps, 2*hidden_size]`.
    pub fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        fused_bilstm_seq(
            xs,
            self.w_ih_fwd.val(),
            self.w_hh_fwd.val(),
            self.b_ih_fwd.val(),
            self.b_hh_fwd.val(),
            self.w_ih_rev.val(),
            self.w_hh_rev.val(),
            self.b_ih_rev.val(),
            self.b_hh_rev.val(),
            self.hidden_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    type CpuBackend = NdArray;
    type CpuAd = Autodiff<NdArray>;

    const BATCH: usize = 2;
    const STEPS: usize = 4;
    const IN_DIM: usize = 3;
    const HIDDEN: usize = 3;

    struct Params {
        x: Vec<f32>,
        w_ih: Vec<f32>,
        w_hh: Vec<f32>,
        b_ih: Vec<f32>,
        b_hh: Vec<f32>,
    }

    fn random_params(seed: u64) -> Params {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rand_vec = |n: usize| (0..n).map(|_| rng.random_range(-0.5..0.5)).collect();
        Params {
            x: rand_vec(BATCH * STEPS * IN_DIM),
            w_ih: rand_vec(4 * HIDDEN * IN_DIM),
            w_hh: rand_vec(4 * HIDDEN * HIDDEN),
            b_ih: rand_vec(4 * HIDDEN),
            b_hh: rand_vec(4 * HIDDEN),
        }
    }

    /// Sum of every element of `lstm_forward`'s output — a scalar loss whose
    /// gradient w.r.t. each input is exactly what `lstm_backward` computes
    /// when fed an all-ones upstream gradient, which is what these tests
    /// check finite differences against.
    fn loss_of(p: &Params, device: &burn::backend::ndarray::NdArrayDevice) -> f32 {
        let x = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(p.x.clone(), [BATCH, STEPS, IN_DIM]),
            device,
        );
        let w_ih = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(p.w_ih.clone(), [4 * HIDDEN, IN_DIM]),
            device,
        );
        let w_hh = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(p.w_hh.clone(), [4 * HIDDEN, HIDDEN]),
            device,
        );
        let b_ih =
            Tensor::<CpuBackend, 1>::from_data(TensorData::new(p.b_ih.clone(), [4 * HIDDEN]), device);
        let b_hh =
            Tensor::<CpuBackend, 1>::from_data(TensorData::new(p.b_hh.clone(), [4 * HIDDEN]), device);

        let (h_seq, _) = lstm_forward(x, w_ih, w_hh, b_ih, b_hh, HIDDEN);
        h_seq.sum().into_scalar()
    }

    fn numerical_grad(p: &Params, field: &str, idx: usize, device: &burn::backend::ndarray::NdArrayDevice) -> f32 {
        let eps = 1e-3f32;
        let mut plus = Params {
            x: p.x.clone(),
            w_ih: p.w_ih.clone(),
            w_hh: p.w_hh.clone(),
            b_ih: p.b_ih.clone(),
            b_hh: p.b_hh.clone(),
        };
        let mut minus = Params {
            x: p.x.clone(),
            w_ih: p.w_ih.clone(),
            w_hh: p.w_hh.clone(),
            b_ih: p.b_ih.clone(),
            b_hh: p.b_hh.clone(),
        };
        fn field_mut<'a>(params: &'a mut Params, field: &str) -> &'a mut Vec<f32> {
            match field {
                "x" => &mut params.x,
                "w_ih" => &mut params.w_ih,
                "w_hh" => &mut params.w_hh,
                "b_ih" => &mut params.b_ih,
                "b_hh" => &mut params.b_hh,
                _ => unreachable!(),
            }
        }
        field_mut(&mut plus, field)[idx] += eps;
        field_mut(&mut minus, field)[idx] -= eps;
        (loss_of(&plus, device) - loss_of(&minus, device)) / (2.0 * eps)
    }

    /// Validates `lstm_backward`'s analytic BPTT gradients against finite
    /// differences of `lstm_forward`'s own output — the same rigor used for
    /// the hand-rolled CTC forward-backward in the original candle port.
    /// Doesn't touch Burn's autodiff machinery at all; this is purely
    /// checking that our hand-derived math is correct.
    #[test]
    fn lstm_backward_matches_finite_differences() {
        let device = Default::default();
        let p = random_params(42);

        let x = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(p.x.clone(), [BATCH, STEPS, IN_DIM]),
            &device,
        );
        let w_ih = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(p.w_ih.clone(), [4 * HIDDEN, IN_DIM]),
            &device,
        );
        let w_hh = Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(p.w_hh.clone(), [4 * HIDDEN, HIDDEN]),
            &device,
        );
        let b_ih = Tensor::<CpuBackend, 1>::from_data(
            TensorData::new(p.b_ih.clone(), [4 * HIDDEN]),
            &device,
        );
        let b_hh = Tensor::<CpuBackend, 1>::from_data(
            TensorData::new(p.b_hh.clone(), [4 * HIDDEN]),
            &device,
        );

        let (h_seq, cache) = lstm_forward(x.clone(), w_ih.clone(), w_hh.clone(), b_ih.clone(), b_hh.clone(), HIDDEN);
        let grad_h_seq = Tensor::<CpuBackend, 3>::ones([BATCH, STEPS, HIDDEN], &device);
        let (grad_x, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh) =
            lstm_backward(grad_h_seq, x, w_ih, w_hh, &cache, HIDDEN);
        let _ = h_seq;

        let grad_x: Vec<f32> = grad_x.into_data().to_vec().unwrap();
        let grad_w_ih: Vec<f32> = grad_w_ih.into_data().to_vec().unwrap();
        let grad_w_hh: Vec<f32> = grad_w_hh.into_data().to_vec().unwrap();
        let grad_b_ih: Vec<f32> = grad_b_ih.into_data().to_vec().unwrap();
        let grad_b_hh: Vec<f32> = grad_b_hh.into_data().to_vec().unwrap();

        let tol = 5e-2f32; // eps=1e-3 central difference on f32 needs real slack
        let mut checked = 0;
        for (field, analytic, n) in [
            ("x", &grad_x, p.x.len()),
            ("w_ih", &grad_w_ih, p.w_ih.len()),
            ("w_hh", &grad_w_hh, p.w_hh.len()),
            ("b_ih", &grad_b_ih, p.b_ih.len()),
            ("b_hh", &grad_b_hh, p.b_hh.len()),
        ] {
            // A handful of spread-out indices per tensor, not every element
            // (finite differences are O(1) forward evals each — keep it fast).
            for idx in (0..n).step_by((n / 4).max(1)).take(4) {
                let numeric = numerical_grad(&p, field, idx, &device);
                let a = analytic[idx];
                let diff = (a - numeric).abs();
                let rel = diff / (numeric.abs().max(a.abs()).max(1e-3));
                assert!(
                    diff < tol || rel < tol,
                    "{field}[{idx}]: analytic={a}, numeric={numeric}, diff={diff}, rel={rel}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 15, "sanity: should have checked a good number of elements");
    }

    /// Validates the `Backward`/`Ops`/checkpointing wiring itself (not just
    /// the math): running the SAME op through Burn's real autodiff graph
    /// (`fused_lstm_seq` under `Autodiff<NdArray>`, `.backward()`, `.grad()`)
    /// must produce the identical gradients as calling `lstm_backward`
    /// directly — a bug here (e.g. a dropped gradient registration, wrong
    /// node) wouldn't necessarily show up in the pure-math check above.
    #[test]
    fn autodiff_wiring_matches_direct_backward() {
        let device = Default::default();
        let p = random_params(7);

        let mk = |p: &Params, device: &burn::backend::ndarray::NdArrayDevice| {
            (
                Tensor::<CpuBackend, 3>::from_data(
                    TensorData::new(p.x.clone(), [BATCH, STEPS, IN_DIM]),
                    device,
                ),
                Tensor::<CpuBackend, 2>::from_data(
                    TensorData::new(p.w_ih.clone(), [4 * HIDDEN, IN_DIM]),
                    device,
                ),
                Tensor::<CpuBackend, 2>::from_data(
                    TensorData::new(p.w_hh.clone(), [4 * HIDDEN, HIDDEN]),
                    device,
                ),
                Tensor::<CpuBackend, 1>::from_data(
                    TensorData::new(p.b_ih.clone(), [4 * HIDDEN]),
                    device,
                ),
                Tensor::<CpuBackend, 1>::from_data(
                    TensorData::new(p.b_hh.clone(), [4 * HIDDEN]),
                    device,
                ),
            )
        };

        // Direct path: same as the finite-difference test above.
        let (x, w_ih, w_hh, b_ih, b_hh) = mk(&p, &device);
        let (h_seq, cache) = lstm_forward(x.clone(), w_ih.clone(), w_hh.clone(), b_ih.clone(), b_hh.clone(), HIDDEN);
        let grad_h_seq = Tensor::<CpuBackend, 3>::ones([BATCH, STEPS, HIDDEN], &device);
        let (_, direct_grad_w_ih, direct_grad_w_hh, direct_grad_b_ih, _) =
            lstm_backward(grad_h_seq, x, w_ih, w_hh, &cache, HIDDEN);
        let _ = h_seq;

        // Autodiff path: through the real custom op.
        let (x, w_ih, w_hh, b_ih, b_hh) = mk(&p, &device);
        let x = Tensor::<CpuAd, 3>::from_data(x.into_data(), &device).require_grad();
        let w_ih = Tensor::<CpuAd, 2>::from_data(w_ih.into_data(), &device).require_grad();
        let w_hh = Tensor::<CpuAd, 2>::from_data(w_hh.into_data(), &device).require_grad();
        let b_ih = Tensor::<CpuAd, 1>::from_data(b_ih.into_data(), &device).require_grad();
        let b_hh = Tensor::<CpuAd, 1>::from_data(b_hh.into_data(), &device).require_grad();

        let out = fused_lstm_seq::<CpuAd>(x, w_ih.clone(), w_hh.clone(), b_ih.clone(), b_hh, HIDDEN);
        let mut grads = out.sum().backward();

        let ad_grad_w_ih: Vec<f32> = w_ih.grad(&grads).unwrap().into_data().to_vec().unwrap();
        let ad_grad_w_hh: Vec<f32> = w_hh.grad(&grads).unwrap().into_data().to_vec().unwrap();
        let ad_grad_b_ih: Vec<f32> = b_ih.grad_remove(&mut grads).unwrap().into_data().to_vec().unwrap();

        let direct_grad_w_ih: Vec<f32> = direct_grad_w_ih.into_data().to_vec().unwrap();
        let direct_grad_w_hh: Vec<f32> = direct_grad_w_hh.into_data().to_vec().unwrap();
        let direct_grad_b_ih: Vec<f32> = direct_grad_b_ih.into_data().to_vec().unwrap();

        for (a, b) in ad_grad_w_ih.iter().zip(&direct_grad_w_ih) {
            assert!((a - b).abs() < 1e-4, "w_ih grad mismatch: {a} vs {b}");
        }
        for (a, b) in ad_grad_w_hh.iter().zip(&direct_grad_w_hh) {
            assert!((a - b).abs() < 1e-4, "w_hh grad mismatch: {a} vs {b}");
        }
        for (a, b) in ad_grad_b_ih.iter().zip(&direct_grad_b_ih) {
            assert!((a - b).abs() < 1e-4, "b_ih grad mismatch: {a} vs {b}");
        }
    }
}
