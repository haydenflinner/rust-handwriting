//! The recognizer, loaded once at startup from the pretrained checkpoint.

use bevy::prelude::*;

use hwr_model::Recognizer;

/// Pretrained on armrest's bundled `data/inks/*.txt` corpus — see
/// `hwr-model/src/bin/train.rs`. Embedded so the app doesn't depend on a
/// runtime file path (matching how armrest embeds `english_ascii.tflite`).
/// Burn's Named-MessagePack format (see `train::save`) — not the old
/// candle-era `model.safetensors`, stale since the Burn migration.
static CHECKPOINT: &[u8] = include_bytes!("../../checkpoints/model.mpk");

/// `NonSend` because candle's CPU tensors aren't required to be `Sync`, and
/// we only ever touch the recognizer from the main thread anyway.
pub struct OcrRecognizer(pub Recognizer);

pub struct OcrPlugin;

impl Plugin for OcrPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_recognizer);
    }
}

fn setup_recognizer(world: &mut World) {
    let recognizer = Recognizer::from_bytes(CHECKPOINT).expect("failed to load recognizer");
    world.insert_non_send(OcrRecognizer(recognizer));
}
