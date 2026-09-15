//! High-level "ink in, text out" API tying together spline encoding, the
//! Burn model, and CTC decoding.

use burn::backend::wgpu::WgpuDevice;
use burn::module::Module;
use burn::record::{
    FullPrecisionSettings, NamedMpkBytesRecorder, NamedMpkFileRecorder, Recorder, RecorderError,
};
use burn::tensor::{Tensor, TensorData};
use hwr_ink::ink::Ink;

use crate::decode::{self, ModelOutput};
use crate::model;
use crate::spline;
use crate::Backend;

#[derive(Debug)]
pub enum Error {
    Record(RecorderError),
}

impl From<RecorderError> for Error {
    fn from(err: RecorderError) -> Self {
        Error::Record(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Record(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

pub struct Recognizer {
    model: model::Recognizer<Backend>,
    device: WgpuDevice,
}

impl Recognizer {
    /// A recognizer with random (untrained) weights — structurally correct,
    /// useful for exercising the pipeline before a trained checkpoint exists.
    pub fn random() -> Result<Self, Error> {
        let device = Default::default();
        let model = model::Recognizer::new(model::Config::default(), &device);
        Ok(Recognizer { model, device })
    }

    /// Load a recognizer from a checkpoint on disk (Burn's Named-MessagePack
    /// format — see `crate::train`'s `save`).
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let device = Default::default();
        let model = model::Recognizer::new(model::Config::default(), &device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model = model.load_file(path.as_ref().to_path_buf(), &recorder, &device)?;
        Ok(Recognizer { model, device })
    }

    /// Load a recognizer from an in-memory checkpoint, e.g. one embedded
    /// with `include_bytes!` so the app doesn't depend on a runtime file
    /// path (matching how armrest embeds `english_ascii.tflite`).
    pub fn from_bytes(data: &[u8]) -> Result<Self, Error> {
        let device = Default::default();
        let model = model::Recognizer::new(model::Config::default(), &device);
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
        let record = recorder.load(data.to_vec(), &device)?;
        let model = model.load_record(record);
        Ok(Recognizer { model, device })
    }

    pub fn recognize<O: ModelOutput>(&self, ink: &Ink, decoder: &O) -> Result<O::Out, Error> {
        let encoded = spline::encode_vec(ink);
        let steps = encoded.len() / spline::WIDTH;

        if steps == 0 {
            return Ok(decoder.read_from(&[]));
        }

        // Replicate to batch=16: CubeCL's GPU autotuner for the TCN
        // front-end's `Conv1d` (see `model.rs`) crashes hard at
        // batch_size=1 — every candidate kernel implementation errors out
        // ("Communication channel with the server is down"). Training's
        // batch=16 never hit this, so single-sample inference is padded
        // out to that exact, already-validated shape rather than a
        // smaller untested one, and the (identical) extra rows are
        // discarded afterward. Wastes ~16x compute for one recognition
        // call, acceptable for a model this small run interactively.
        const BATCH: usize = 16;
        let mut batched = Vec::with_capacity(encoded.len() * BATCH);
        for _ in 0..BATCH {
            batched.extend_from_slice(&encoded);
        }

        let input = Tensor::<Backend, 3>::from_data(
            TensorData::new(batched, [BATCH, steps, spline::WIDTH]),
            &self.device,
        );
        let output = self.model.forward(input);
        let flat: Vec<f32> = output
            .into_data()
            .to_vec()
            .expect("f32 tensor data should convert to Vec<f32>");

        let per_sample = flat.len() / BATCH;
        Ok(decoder.read_from(&flat[..per_sample]))
    }

    pub fn recognize_greedy(&self, ink: &Ink) -> Result<String, Error> {
        self.recognize(ink, &decode::Greedy)
    }
}
