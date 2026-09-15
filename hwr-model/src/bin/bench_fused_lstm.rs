//! Benchmark: `hwr_model::fused_lstm`'s custom op vs. stock `burn::nn::BiLstm`
//! for a single BiLSTM layer's forward+backward, at increasing sequence
//! length — mirrors the diagnostic in `burn-spike` that first showed stock
//! `BiLstm` scaling super-linearly with T under Burn's autodiff. Run on the
//! production `Backend` (Wgpu) since that's what actually matters.

use std::time::Instant;

use burn::nn::{BiLstm, BiLstmConfig};
use burn::tensor::{Distribution, Tensor};

use hwr_model::fused_lstm::fused_bilstm_seq;
use hwr_model::TrainBackend;

const BATCH: usize = 16;
const INPUT_WIDTH: usize = 4; // matches hwr_model::spline::WIDTH
const HIDDEN: usize = 64;

fn bench_stock(device: &burn::backend::wgpu::WgpuDevice, steps: usize) -> f64 {
    let layer: BiLstm<TrainBackend> = BiLstmConfig::new(INPUT_WIDTH, HIDDEN, true).init(device);
    let run = || {
        let input = Tensor::<TrainBackend, 3>::random(
            [BATCH, steps, INPUT_WIDTH],
            Distribution::Default,
            device,
        );
        let (out, _) = layer.forward(input, None);
        let loss = out.sum();
        loss.backward();
    };
    // Warm-up: pays for one-time GPU shader compilation at this shape, so
    // the timed run below measures steady-state (repeated-batch) cost, not
    // a one-off JIT cost that a real multi-epoch training run amortizes
    // away across thousands of batches.
    run();
    let start = Instant::now();
    run();
    start.elapsed().as_secs_f64()
}

fn bench_fused(device: &burn::backend::wgpu::WgpuDevice, steps: usize) -> f64 {
    // Random weights with the same shapes our fused op expects (4H, in) /
    // (4H, H) / (4H,), for both directions.
    let mk2 = |r: usize, c: usize| {
        Tensor::<TrainBackend, 2>::random([r, c], Distribution::Default, device).require_grad()
    };
    let mk1 =
        |n: usize| Tensor::<TrainBackend, 1>::random([n], Distribution::Default, device).require_grad();

    let (w_ih_f, w_hh_f, b_ih_f, b_hh_f) = (
        mk2(4 * HIDDEN, INPUT_WIDTH),
        mk2(4 * HIDDEN, HIDDEN),
        mk1(4 * HIDDEN),
        mk1(4 * HIDDEN),
    );
    let (w_ih_r, w_hh_r, b_ih_r, b_hh_r) = (
        mk2(4 * HIDDEN, INPUT_WIDTH),
        mk2(4 * HIDDEN, HIDDEN),
        mk1(4 * HIDDEN),
        mk1(4 * HIDDEN),
    );

    let run = || {
        let input = Tensor::<TrainBackend, 3>::random(
            [BATCH, steps, INPUT_WIDTH],
            Distribution::Default,
            device,
        );
        let out = fused_bilstm_seq(
            input,
            w_ih_f.clone(),
            w_hh_f.clone(),
            b_ih_f.clone(),
            b_hh_f.clone(),
            w_ih_r.clone(),
            w_hh_r.clone(),
            b_ih_r.clone(),
            b_hh_r.clone(),
            HIDDEN,
        );
        let loss = out.sum();
        loss.backward();
    };
    run(); // warm-up, see bench_stock
    let start = Instant::now();
    run();
    start.elapsed().as_secs_f64()
}

fn main() {
    let device = Default::default();
    let step_counts = [20usize, 80, 320, 1280];

    println!("steps    stock BiLstm    fused op    speedup");
    for &steps in &step_counts {
        let stock = bench_stock(&device, steps);
        let fused = bench_fused(&device, steps);
        println!(
            "{steps:>5}    {stock:>10.3}s    {fused:>7.3}s    {:>6.1}x",
            stock / fused.max(1e-9)
        );
    }
}
