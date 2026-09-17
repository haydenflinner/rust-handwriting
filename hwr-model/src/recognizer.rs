//! High-level "ink in, text out" API tying together spline encoding, the
//! Burn model, and CTC decoding.

use burn::backend::wgpu::WgpuDevice;
use burn::module::Module;
use burn::record::{
    FullPrecisionSettings, NamedMpkBytesRecorder, NamedMpkFileRecorder, Recorder, RecorderError,
};
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
        let encoded = spline::encode_strokes(ink);
        let steps = encoded.len() / spline::STROKE_DIM;

        if steps == 0 {
            return Ok(decoder.read_from(&[]));
        }

        let (strokes, images, pad) =
            spline::pack_hat_batch([(encoded.as_slice(), steps)], 1, 1, steps);
        let (strokes, images, pad_mask) =
            model::packed_inputs(strokes, images, pad, 1, steps, &self.device);
        let output = self.model.forward(strokes, images, Some(pad_mask));
        let flat: Vec<f32> = output
            .into_data()
            .to_vec()
            .expect("f32 tensor data should convert to Vec<f32>");

        Ok(decoder.read_from(&flat))
    }

    pub fn recognize_greedy(&self, ink: &Ink) -> Result<String, Error> {
        self.recognize(ink, &decode::Greedy)
    }
}
