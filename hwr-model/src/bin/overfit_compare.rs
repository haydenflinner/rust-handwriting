//! One-off diagnostic: does the SAME architecture, using Burn's stock
//! (trusted) `nn::BiLstm` instead of our custom `FusedBiLstmLayer`, escape
//! the "predict blank everywhere" CTC collapse on a tiny 13-sample overfit
//! test where the fused version got stuck at train_cer=1.0 for 57 straight
//! epochs despite loss dropping 20x? If stock also gets stuck, the bug (or
//! non-bug) is architecture/data/hyperparameter-general, not specific to
//! our hand-derived BPTT gradients. If stock escapes it, our fused
//! implementation likely has a real bug.
//!
//! Usage: overfit_compare [CORPUS] [EPOCHS]

use burn::module::Module;
use burn::nn::loss::{CTCLossConfig, Reduction};
use burn::nn::{BiLstm, BiLstmConfig, Dropout, DropoutConfig, Linear, LinearConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

use hwr_model::{decode, spline, Backend as ProdBackend, TrainBackend};

#[derive(Module, Debug)]
struct StockRecognizer<B: Backend> {
    layers: Vec<BiLstm<B>>,
    dropout: Dropout,
    dense: Linear<B>,
}

impl<B: Backend> StockRecognizer<B> {
    fn new(input_width: usize, hidden: usize, num_layers: usize, classes: usize, device: &B::Device) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        let mut in_dim = input_width;
        for _ in 0..num_layers {
            layers.push(BiLstmConfig::new(in_dim, hidden, true).init(device));
            in_dim = hidden * 2;
        }
        StockRecognizer {
            layers,
            dropout: DropoutConfig::new(0.5).init(),
            dense: LinearConfig::new(in_dim, classes).init(device),
        }
    }

    fn forward_logits(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = xs;
        for layer in &self.layers {
            let (out, _) = layer.forward(h, None);
            h = self.dropout.forward(out);
        }
        self.dense.forward(h)
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let corpus_path = args.next().unwrap_or_else(|| "/tmp/overfit_test.txt".to_string());
    let epochs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(60);

    let pairs = hwr_model::corpus::load_pairs(&corpus_path).expect("failed to load corpus");
    println!("Loaded {} pairs from {corpus_path}", pairs.len());

    let device = Default::default();
    let mut model: StockRecognizer<TrainBackend> =
        StockRecognizer::new(spline::WIDTH, 64, 5, decode::classes(), &device);

    let mut optimizer = AdamWConfig::new().init();
    let blank = decode::classes() - 1;
    let ctc = CTCLossConfig::new().with_blank(blank).init();

    struct Sample {
        labels: Vec<usize>,
        encoded: Vec<f32>,
        steps: usize,
    }
    let samples: Vec<Sample> = pairs
        .iter()
        .filter_map(|(text, ink)| {
            let labels = decode::encode_labels(text)?;
            if labels.is_empty() {
                return None;
            }
            let encoded = spline::encode_vec(ink);
            let steps = encoded.len() / spline::WIDTH;
            if steps == 0 || steps < 2 * labels.len() + 1 {
                return None;
            }
            Some(Sample { labels, encoded, steps })
        })
        .collect();
    println!("{} usable samples after CTC-feasibility filter", samples.len());

    let b = samples.len();
    let max_steps = samples.iter().map(|s| s.steps).max().unwrap();
    let max_target_len = samples.iter().map(|s| s.labels.len()).max().unwrap();

    let mut input_buf = vec![0f32; b * max_steps * spline::WIDTH];
    let mut target_buf = vec![0i32; b * max_target_len];
    let mut input_lengths = vec![0i32; b];
    let mut target_lengths = vec![0i32; b];
    for (bi, s) in samples.iter().enumerate() {
        let dst = bi * max_steps * spline::WIDTH;
        input_buf[dst..dst + s.steps * spline::WIDTH].copy_from_slice(&s.encoded);
        input_lengths[bi] = s.steps as i32;
        let tdst = bi * max_target_len;
        for (j, &l) in s.labels.iter().enumerate() {
            target_buf[tdst + j] = l as i32;
        }
        target_lengths[bi] = s.labels.len() as i32;
    }

    for epoch in 0..epochs {
        let input = Tensor::<TrainBackend, 3>::from_data(
            TensorData::new(input_buf.clone(), [b, max_steps, spline::WIDTH]),
            &device,
        );
        let logits = model.forward_logits(input);
        let log_probs = log_softmax(logits, 2).swap_dims(0, 1);
        let targets = Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(target_buf.clone(), [b, max_target_len]),
            &device,
        );
        let in_lens = Tensor::<TrainBackend, 1, Int>::from_data(
            TensorData::new(input_lengths.clone(), [b]),
            &device,
        );
        let tgt_lens = Tensor::<TrainBackend, 1, Int>::from_data(
            TensorData::new(target_lengths.clone(), [b]),
            &device,
        );
        let loss = ctc.forward_with_reduction(log_probs, targets, in_lens, tgt_lens, Reduction::Mean);
        let loss_val: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(1e-3, model, grads);

        if epoch % 5 == 0 || epoch == epochs - 1 {
            // Inference pass (no autodiff) to compute CER, mirroring eval::mean_cer.
            let inference_input = Tensor::<ProdBackend, 3>::from_data(
                TensorData::new(input_buf.clone(), [b, max_steps, spline::WIDTH]),
                &device,
            );
            // Rebuild an inference-mode copy via valid() is awkward for a local
            // struct without AutodiffModule wiring here, so just run forward
            // with dropout still active (module is TrainBackend); good enough
            // to see whether it's learning at all, not for a final metric.
            let _ = inference_input; // (kept for clarity; using train-mode forward below)
            let probs_input = Tensor::<TrainBackend, 3>::from_data(
                TensorData::new(input_buf.clone(), [b, max_steps, spline::WIDTH]),
                &device,
            );
            let probs = burn::tensor::activation::softmax(model.forward_logits(probs_input), 2);
            let flat: Vec<f32> = probs.into_data().to_vec().unwrap();
            let per_sample_floats = flat.len() / b;
            let mut total_cer = 0.0f64;
            for (bi, s) in samples.iter().enumerate() {
                let sample_probs = &flat[bi * per_sample_floats..(bi + 1) * per_sample_floats];
                let decoded = decode::greedy_decode(sample_probs);
                let expected: String = s.labels.iter().map(|&l| decode::CHARS.as_bytes()[l] as char).collect();
                let dist = strsim::levenshtein(&expected, &decoded) as f64;
                let cer = if expected.is_empty() { 0.0 } else { dist / expected.chars().count() as f64 };
                total_cer += cer;
                if epoch == epochs - 1 {
                    println!("  {expected:?} -> {decoded:?}");
                }
            }
            println!(
                "epoch {epoch:>3}: loss={loss_val:.4} cer={:.4}",
                total_cer / samples.len() as f64
            );
        }
    }
}
