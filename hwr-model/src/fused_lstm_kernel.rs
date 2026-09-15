//! A genuine on-GPU fused LSTM kernel: the *entire* per-timestep sequential
//! recurrence for one LSTM direction runs inside a single CubeCL kernel
//! launch (one GPU "cube" per batch row, `sync_cube()`-fenced phases, a
//! runtime-length time loop on-device), instead of `fused_lstm.rs`'s
//! `lstm_forward`/`lstm_backward` composing ordinary tensor ops in a
//! Rust-level `for t in 0..steps` loop.
//!
//! Why this exists on top of that: `fused_lstm.rs` already collapsed
//! Burn's autodiff *graph-node* count from O(T) to O(1) per layer (see its
//! module docs), which fixed the CPU-side graph-bookkeeping cost we
//! originally diagnosed. But profiling the live training process afterward
//! (macOS `sample`) still showed real wall-clock time going into
//! `burn_fusion`'s `GlobalFusionClient::register`/`submit_inner`, with
//! threads blocking on `nanosleep`/`__semwait_signal` — genuine GPU-dispatch
//! queue backpressure. The autodiff-node fix didn't touch that: `lstm_seq`'s
//! internal per-timestep loop still issued one GPU dispatch per op per
//! timestep (matmul, cat, elementwise gate math), so *dispatch* count was
//! still O(T) even though *graph-node* count was O(1). This kernel collapses
//! dispatch count to O(1) too, by moving the sequential loop itself onto the
//! device.
//!
//! Modeled directly on `burn_cubecl::kernel::ctc::ctc_loss_kernel`, which
//! already runs CTC's own sequential alpha/beta recursion the same way —
//! this is a trusted, production pattern within Burn itself, not a novel
//! technique.
//!
//! ## The fusion tradeoff (read before touching Cargo.toml)
//! A hand-written kernel needs a raw `CubeTensor` to launch against. But
//! production `Wgpu` is `Fusion<CubeBackend<...>>` by default (burn-wgpu's
//! own `default` feature list includes `fusion`) — under fusion, a
//! `Tensor<Wgpu, _>`'s primitive is a `FusionTensor`, which defers and
//! coalesces small composed ops rather than dispatching them immediately,
//! and isn't something a downstream crate can unwrap into a `CubeTensor`
//! without patching Burn itself (same category of problem as the
//! `topological_sort` fix, i.e. not something to take on here). The
//! workaround, applied in `hwr-model/Cargo.toml`: disable Burn's default
//! features and explicitly re-enable only `["std", "wgpu", "autodiff"]` —
//! no `"fusion"` — which makes `Wgpu` = bare `CubeBackend`, so
//! `FloatTensor<Wgpu>` really is a `CubeTensor`.
//!
//! This is a real, non-free trade-off, empirically measured: composed
//! tensor-op code (stock `BiLstm`, or any hot path *not* converted to a
//! kernel) gets dramatically SLOWER without fusion (stock `BiLstm` at
//! steps=320 went from ~1s to over 6 minutes) — fusion was doing real
//! op-coalescing work elsewhere. It's a net win here specifically because
//! *both* `lstm_seq_forward_cached` and `lstm_seq_backward` (this module)
//! are now full kernels with nothing left in the LSTM's hot path relying on
//! fusion. Everything else in `Recognizer` (`BatchNorm`, `Dropout`,
//! `Linear`, `CTCLoss`) is either a single op or (for `CTCLoss`) already
//! non-fuseable by Burn's own design — but this was validated by direct
//! benchmark against the running corpus, not assumed; see the commit/PR
//! history around when this module landed for that check.

use burn::tensor::{Tensor, TensorMetadata, TensorPrimitive};
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use crate::fused_lstm::LstmCache;
use crate::Backend as ProdBackend;

fn to_cube3(t: Tensor<ProdBackend, 3>) -> CubeTensor<WgpuRuntime> {
    t.into_primitive().tensor()
}
fn to_cube2(t: Tensor<ProdBackend, 2>) -> CubeTensor<WgpuRuntime> {
    t.into_primitive().tensor()
}
fn to_cube1(t: Tensor<ProdBackend, 1>) -> CubeTensor<WgpuRuntime> {
    t.into_primitive().tensor()
}
fn from_cube3(t: CubeTensor<WgpuRuntime>) -> Tensor<ProdBackend, 3> {
    Tensor::from_primitive(TensorPrimitive::Float(t))
}
fn from_cube2(t: CubeTensor<WgpuRuntime>) -> Tensor<ProdBackend, 2> {
    Tensor::from_primitive(TensorPrimitive::Float(t))
}

/// One cube per batch row. Threads stride over the `4*hidden` gate-unit
/// index space to compute both matmuls' dot products (phase 1), sync, then
/// stride over the `hidden` index space to combine gates into new c/h and
/// publish them (phase 2) — same two-phase-with-`sync_cube` pattern as
/// `ctc_loss_kernel`'s alpha recursion, just with LSTM gate math instead of
/// the CTC forward recursion. Also publishes the per-timestep activated
/// gate values (`cache_*_out`) that `lstm_backward_kernel` needs for BPTT.
#[cube(launch)]
fn lstm_forward_kernel<F: Float>(
    x: &cubecl::prelude::Tensor<F>,         // [B, T, Din]
    w_ih: &cubecl::prelude::Tensor<F>,      // [4H, Din]
    w_hh: &cubecl::prelude::Tensor<F>,      // [4H, H]
    b_ih: &cubecl::prelude::Tensor<F>,      // [4H]
    b_hh: &cubecl::prelude::Tensor<F>,      // [4H]
    h_out: &mut cubecl::prelude::Tensor<F>, // [B, T, H]
    cache_i_out: &mut cubecl::prelude::Tensor<F>,
    cache_f_out: &mut cubecl::prelude::Tensor<F>,
    cache_g_out: &mut cubecl::prelude::Tensor<F>,
    cache_o_out: &mut cubecl::prelude::Tensor<F>,
    cache_c_out: &mut cubecl::prelude::Tensor<F>,
    #[comptime] hidden: usize,
    #[comptime] in_dim: usize,
) {
    let b = CUBE_POS_X as usize;
    let cube_dim = CUBE_DIM_X as usize;
    let four_h = 4 * hidden;

    let steps = x.shape(1);
    let x_t_stride = x.stride(0);
    let x_s_stride = x.stride(1);
    let wih_j_stride = w_ih.stride(0);
    let whh_j_stride = w_hh.stride(0);
    let ho_b_stride = h_out.stride(0);
    let ho_s_stride = h_out.stride(1);

    // Shared: gates[0..4H] holds this timestep's ACTIVATED i,f,g,o values
    // (quadrant j/hidden selects which); h_state/c_state[0..H] persist the
    // recurrent state across timesteps within this cube.
    let mut gates = SharedMemory::<F>::new(four_h);
    let mut h_state = SharedMemory::<F>::new(hidden);
    let mut c_state = SharedMemory::<F>::new(hidden);

    let mut idx = UNIT_POS_X as usize;
    while idx < hidden {
        h_state[idx] = F::new(0.0_f32);
        c_state[idx] = F::new(0.0_f32);
        idx += cube_dim;
    }
    sync_cube();

    for t in 0..steps {
        // Phase 1: each thread owns a strided subset of the 4H gate units,
        // computes its pre-activation via two dot products, activates it.
        let mut j = UNIT_POS_X as usize;
        while j < four_h {
            let mut acc = b_ih[j] + b_hh[j];
            let mut k = 0usize;
            while k < in_dim {
                acc += x[b * x_t_stride + t * x_s_stride + k] * w_ih[j * wih_j_stride + k];
                k += 1;
            }
            let mut k = 0usize;
            while k < hidden {
                acc += h_state[k] * w_hh[j * whh_j_stride + k];
                k += 1;
            }

            let quadrant = j / hidden;
            let activated = if quadrant == 2 {
                F::tanh(acc)
            } else {
                F::new(1.0_f32) / (F::new(1.0_f32) + F::exp(F::new(0.0_f32) - acc))
            };
            gates[j] = activated;
            j += cube_dim;
        }
        sync_cube();

        // Phase 2: each thread owns a strided subset of the H hidden units,
        // combines that unit's four cached gate values into new c/h, and
        // publishes both the new recurrent state and this timestep's output.
        let mut h_idx = UNIT_POS_X as usize;
        while h_idx < hidden {
            let i_g = gates[h_idx];
            let f_g = gates[hidden + h_idx];
            let g_g = gates[2 * hidden + h_idx];
            let o_g = gates[3 * hidden + h_idx];

            let c_new = f_g * c_state[h_idx] + i_g * g_g;
            let h_new = o_g * F::tanh(c_new);

            c_state[h_idx] = c_new;
            h_state[h_idx] = h_new;
            let out_idx = b * ho_b_stride + t * ho_s_stride + h_idx;
            h_out[out_idx] = h_new;
            cache_i_out[out_idx] = i_g;
            cache_f_out[out_idx] = f_g;
            cache_g_out[out_idx] = g_g;
            cache_o_out[out_idx] = o_g;
            cache_c_out[out_idx] = c_new;

            h_idx += cube_dim;
        }
        sync_cube();
    }
}

/// One cube per batch row, reverse sequential time loop. Six phases per
/// timestep, each a strided pass over a different index space with a
/// `sync_cube()` between: (A) per-hidden-unit BPTT math -> `grad_gates[4H]`
/// + updates `grad_c_next` in place (owned per-index, no cross-thread
/// hazard); (B) `grad_x[t,:]` via `grad_gates . w_ih`; (C) new
/// `grad_h_next` via `grad_gates . w_hh` (must follow a full `grad_gates`
/// sync — depends on every j, not just one thread's own indices); (D/E/F)
/// weight/bias gradient accumulation via global-memory read-add-write, one
/// accumulator per batch row (summed across the batch afterward on the
/// host — see `kernel_backward`'s `sum_dim(0)` — avoids needing atomics;
/// simple and correct, not register-accumulated, a known further
/// optimization if ever needed).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn lstm_backward_kernel<F: Float>(
    grad_h_seq: &cubecl::prelude::Tensor<F>, // [B,T,H]
    x: &cubecl::prelude::Tensor<F>,          // [B,T,Din]
    w_ih: &cubecl::prelude::Tensor<F>,       // [4H,Din]
    w_hh: &cubecl::prelude::Tensor<F>,       // [4H,H]
    cache_i: &cubecl::prelude::Tensor<F>,
    cache_f: &cubecl::prelude::Tensor<F>,
    cache_g: &cubecl::prelude::Tensor<F>,
    cache_o: &cubecl::prelude::Tensor<F>,
    cache_c: &cubecl::prelude::Tensor<F>,
    cache_h: &cubecl::prelude::Tensor<F>,
    grad_x: &mut cubecl::prelude::Tensor<F>,    // [B,T,Din]
    grad_w_ih: &mut cubecl::prelude::Tensor<F>, // [B,4H,Din]
    grad_w_hh: &mut cubecl::prelude::Tensor<F>, // [B,4H,H]
    grad_b: &mut cubecl::prelude::Tensor<F>,    // [B,4H]
    #[comptime] hidden: usize,
    #[comptime] in_dim: usize,
) {
    let b = CUBE_POS_X as usize;
    let cube_dim = CUBE_DIM_X as usize;
    let four_h = 4 * hidden;
    let steps = x.shape(1);

    let ghs_b = grad_h_seq.stride(0);
    let ghs_s = grad_h_seq.stride(1);
    let x_b = x.stride(0);
    let x_s = x.stride(1);
    let c_b = cache_i.stride(0);
    let c_s = cache_i.stride(1);
    let wih_j = w_ih.stride(0);
    let whh_j = w_hh.stride(0);
    let gx_b = grad_x.stride(0);
    let gx_s = grad_x.stride(1);
    let gwih_b = grad_w_ih.stride(0);
    let gwih_j = grad_w_ih.stride(1);
    let gwhh_b = grad_w_hh.stride(0);
    let gwhh_j = grad_w_hh.stride(1);
    let gb_b = grad_b.stride(0);

    let mut grad_gates = SharedMemory::<F>::new(four_h);
    let mut grad_h_next = SharedMemory::<F>::new(hidden);
    let mut grad_c_next = SharedMemory::<F>::new(hidden);
    let mut h_prev_shared = SharedMemory::<F>::new(hidden);

    let mut idx = UNIT_POS_X as usize;
    while idx < hidden {
        grad_h_next[idx] = F::new(0.0_f32);
        grad_c_next[idx] = F::new(0.0_f32);
        idx += cube_dim;
    }

    // Zero this batch row's weight/bias gradient accumulators once up front.
    let mut idx = UNIT_POS_X as usize;
    while idx < four_h * in_dim {
        grad_w_ih[b * gwih_b + idx] = F::new(0.0_f32);
        idx += cube_dim;
    }
    let mut idx = UNIT_POS_X as usize;
    while idx < four_h * hidden {
        grad_w_hh[b * gwhh_b + idx] = F::new(0.0_f32);
        idx += cube_dim;
    }
    let mut idx = UNIT_POS_X as usize;
    while idx < four_h {
        grad_b[b * gb_b + idx] = F::new(0.0_f32);
        idx += cube_dim;
    }
    sync_cube();

    let mut t_rev = 0usize;
    while t_rev < steps {
        let t = steps - 1 - t_rev;

        // Phase A: per-hidden-unit BPTT math.
        let mut h_idx = UNIT_POS_X as usize;
        while h_idx < hidden {
            let grad_h_t = grad_h_seq[b * ghs_b + t * ghs_s + h_idx] + grad_h_next[h_idx];

            let c_t = cache_c[b * c_b + t * c_s + h_idx];
            let o_t = cache_o[b * c_b + t * c_s + h_idx];
            let i_t = cache_i[b * c_b + t * c_s + h_idx];
            let f_t = cache_f[b * c_b + t * c_s + h_idx];
            let g_t = cache_g[b * c_b + t * c_s + h_idx];
            let mut c_prev_t = F::new(0.0_f32);
            let mut h_prev_t = F::new(0.0_f32);
            if t > 0 {
                c_prev_t = cache_c[b * c_b + (t - 1) * c_s + h_idx];
                h_prev_t = cache_h[b * c_b + (t - 1) * c_s + h_idx];
            }
            h_prev_shared[h_idx] = h_prev_t;

            let tanh_c_t = F::tanh(c_t);
            let grad_o_t = grad_h_t * tanh_c_t;
            let one_minus_tanh2 = F::new(1.0_f32) - tanh_c_t * tanh_c_t;
            let grad_c_t = grad_h_t * o_t * one_minus_tanh2 + grad_c_next[h_idx];

            let grad_f_t = grad_c_t * c_prev_t;
            let grad_i_t = grad_c_t * g_t;
            let grad_g_t = grad_c_t * i_t;
            grad_c_next[h_idx] = grad_c_t * f_t;

            grad_gates[h_idx] = grad_i_t * i_t * (F::new(1.0_f32) - i_t);
            grad_gates[hidden + h_idx] = grad_f_t * f_t * (F::new(1.0_f32) - f_t);
            grad_gates[2 * hidden + h_idx] = grad_g_t * (F::new(1.0_f32) - g_t * g_t);
            grad_gates[3 * hidden + h_idx] = grad_o_t * o_t * (F::new(1.0_f32) - o_t);

            h_idx += cube_dim;
        }
        sync_cube();

        // Phase B: grad_x[t, k] = sum_j grad_gates[j] * w_ih[j, k].
        let mut k = UNIT_POS_X as usize;
        while k < in_dim {
            let mut acc = F::new(0.0_f32);
            let mut j = 0usize;
            while j < four_h {
                acc += grad_gates[j] * w_ih[j * wih_j + k];
                j += 1;
            }
            grad_x[b * gx_b + t * gx_s + k] = acc;
            k += cube_dim;
        }

        // Phase C: new grad_h_next[k] = sum_j grad_gates[j] * w_hh[j, k].
        // Must not alias grad_h_next while other threads are still in Phase
        // A of this same iteration reading the OLD values — the sync_cube
        // after Phase A already guarantees that.
        let mut k = UNIT_POS_X as usize;
        while k < hidden {
            let mut acc = F::new(0.0_f32);
            let mut j = 0usize;
            while j < four_h {
                acc += grad_gates[j] * w_hh[j * whh_j + k];
                j += 1;
            }
            grad_h_next[k] = acc;
            k += cube_dim;
        }
        sync_cube();

        // Phase D: grad_w_ih[b,j,k] += grad_gates[j] * x[b,t,k].
        let mut idx = UNIT_POS_X as usize;
        while idx < four_h * in_dim {
            let j = idx / in_dim;
            let k = idx % in_dim;
            let contrib = grad_gates[j] * x[b * x_b + t * x_s + k];
            grad_w_ih[b * gwih_b + j * gwih_j + k] += contrib;
            idx += cube_dim;
        }

        // Phase E: grad_w_hh[b,j,k] += grad_gates[j] * h_prev[k].
        let mut idx = UNIT_POS_X as usize;
        while idx < four_h * hidden {
            let j = idx / hidden;
            let k = idx % hidden;
            let contrib = grad_gates[j] * h_prev_shared[k];
            grad_w_hh[b * gwhh_b + j * gwhh_j + k] += contrib;
            idx += cube_dim;
        }

        // Phase F: grad_b[b,j] += grad_gates[j].
        let mut j = UNIT_POS_X as usize;
        while j < four_h {
            grad_b[b * gb_b + j] += grad_gates[j];
            j += cube_dim;
        }
        sync_cube();

        t_rev += 1;
    }
}

/// Launch [`lstm_forward_kernel`] and package its outputs into our
/// [`LstmCache`] — the production (`Wgpu`) implementation of
/// `FusedLstm::lstm_seq_forward_cached`.
pub fn kernel_forward(
    x: Tensor<ProdBackend, 3>,
    w_ih: Tensor<ProdBackend, 2>,
    w_hh: Tensor<ProdBackend, 2>,
    b_ih: Tensor<ProdBackend, 1>,
    b_hh: Tensor<ProdBackend, 1>,
    hidden: usize,
) -> (Tensor<ProdBackend, 3>, LstmCache<ProdBackend>) {
    let x = to_cube3(x);
    let w_ih_cube = to_cube2(w_ih);
    let w_hh_cube = to_cube2(w_hh);
    let b_ih_cube = to_cube1(b_ih);
    let b_hh_cube = to_cube1(b_hh);

    let [batch, steps, _in_dim] = x.shape().dims::<3>();
    let in_dim = w_ih_cube.shape().dims::<2>()[1];

    let client = x.client.clone();
    let device = x.device.clone();
    let dtype = x.dtype;

    let alloc = || {
        empty_device_dtype::<WgpuRuntime>(
            client.clone(),
            device.clone(),
            burn::tensor::Shape::new([batch, steps, hidden]),
            dtype,
        )
    };
    let h_out = alloc();
    let cache_i = alloc();
    let cache_f = alloc();
    let cache_g = alloc();
    let cache_o = alloc();
    let cache_c = alloc();

    let cube_dim_x = (4 * hidden as u32).min(256);
    let cube_count = CubeCount::Static(batch as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(cube_dim_x);

    lstm_forward_kernel::launch::<f32, WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        x.into_tensor_arg(),
        w_ih_cube.into_tensor_arg(),
        w_hh_cube.into_tensor_arg(),
        b_ih_cube.into_tensor_arg(),
        b_hh_cube.into_tensor_arg(),
        h_out.clone().into_tensor_arg(),
        cache_i.clone().into_tensor_arg(),
        cache_f.clone().into_tensor_arg(),
        cache_g.clone().into_tensor_arg(),
        cache_o.clone().into_tensor_arg(),
        cache_c.clone().into_tensor_arg(),
        hidden,
        in_dim,
    );

    let h_seq = from_cube3(h_out);
    let cache = LstmCache {
        i: from_cube3(cache_i),
        f: from_cube3(cache_f),
        g: from_cube3(cache_g),
        o: from_cube3(cache_o),
        c: from_cube3(cache_c),
        h: h_seq.clone(),
    };
    (h_seq, cache)
}

/// Launch [`lstm_backward_kernel`] and reduce its per-batch-row partial
/// weight/bias gradients into final `[4H, Din]`/`[4H, H]`/`[4H]` gradients
/// (a single `sum_dim(0)` per tensor — cheap, and avoids needing atomics
/// inside the kernel) — the production (`Wgpu`) implementation of
/// `FusedLstm::lstm_seq_backward`.
pub fn kernel_backward(
    grad_h_seq: Tensor<ProdBackend, 3>,
    x: Tensor<ProdBackend, 3>,
    w_ih: Tensor<ProdBackend, 2>,
    w_hh: Tensor<ProdBackend, 2>,
    cache: &LstmCache<ProdBackend>,
    hidden: usize,
) -> (
    Tensor<ProdBackend, 3>,
    Tensor<ProdBackend, 2>,
    Tensor<ProdBackend, 2>,
    Tensor<ProdBackend, 1>,
    Tensor<ProdBackend, 1>,
) {
    let in_dim = x.dims()[2];
    let [batch, steps, _] = x.dims();
    let four_h = 4 * hidden;

    let grad_h_seq = to_cube3(grad_h_seq);
    let x = to_cube3(x);
    let w_ih_cube = to_cube2(w_ih);
    let w_hh_cube = to_cube2(w_hh);
    let cache_i = to_cube3(cache.i.clone());
    let cache_f = to_cube3(cache.f.clone());
    let cache_g = to_cube3(cache.g.clone());
    let cache_o = to_cube3(cache.o.clone());
    let cache_c = to_cube3(cache.c.clone());
    let cache_h = to_cube3(cache.h.clone());

    let client = x.client.clone();
    let device = x.device.clone();
    let dtype = x.dtype;

    let grad_x_out = empty_device_dtype::<WgpuRuntime>(
        client.clone(),
        device.clone(),
        burn::tensor::Shape::new([batch, steps, in_dim]),
        dtype,
    );
    let grad_w_ih_out = empty_device_dtype::<WgpuRuntime>(
        client.clone(),
        device.clone(),
        burn::tensor::Shape::new([batch, four_h, in_dim]),
        dtype,
    );
    let grad_w_hh_out = empty_device_dtype::<WgpuRuntime>(
        client.clone(),
        device.clone(),
        burn::tensor::Shape::new([batch, four_h, hidden]),
        dtype,
    );
    let grad_b_out = empty_device_dtype::<WgpuRuntime>(
        client.clone(),
        device,
        burn::tensor::Shape::new([batch, four_h]),
        dtype,
    );

    let cube_dim_x = (4 * hidden as u32).min(256);
    let cube_count = CubeCount::Static(batch as u32, 1, 1);
    let cube_dim = CubeDim::new_1d(cube_dim_x);

    lstm_backward_kernel::launch::<f32, WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        grad_h_seq.into_tensor_arg(),
        x.into_tensor_arg(),
        w_ih_cube.into_tensor_arg(),
        w_hh_cube.into_tensor_arg(),
        cache_i.into_tensor_arg(),
        cache_f.into_tensor_arg(),
        cache_g.into_tensor_arg(),
        cache_o.into_tensor_arg(),
        cache_c.into_tensor_arg(),
        cache_h.into_tensor_arg(),
        grad_x_out.clone().into_tensor_arg(),
        grad_w_ih_out.clone().into_tensor_arg(),
        grad_w_hh_out.clone().into_tensor_arg(),
        grad_b_out.clone().into_tensor_arg(),
        hidden,
        in_dim,
    );

    let grad_x = from_cube3(grad_x_out);
    let grad_w_ih = from_cube3(grad_w_ih_out)
        .sum_dim(0)
        .reshape([four_h, in_dim]);
    let grad_w_hh = from_cube3(grad_w_hh_out)
        .sum_dim(0)
        .reshape([four_h, hidden]);
    let grad_b = from_cube2(grad_b_out).sum_dim(0).reshape([four_h]);

    (grad_x, grad_w_ih, grad_w_hh, grad_b.clone(), grad_b)
}
