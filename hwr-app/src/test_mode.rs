//! Test mode: write in the canvas, see what the recognizer decodes.

use bevy::prelude::*;

use hwr_ink::ink::Ink;

use crate::mode::AppMode;
use crate::ocr::{OcrCheckpointSource, OcrRecognizer};
use crate::writing_cell::{make_writable, CellInk};

/// Pause after the last pen-up before calling a VLM, so a word written as
/// several strokes is recognized once instead of once per letter.
const VLM_DEBOUNCE: f32 = 0.45;

pub struct TestModePlugin;

impl Plugin for TestModePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RecognizedText::default())
            .insert_resource(PendingOcr::default())
            .add_systems(Startup, setup_ui)
            .add_systems(OnEnter(AppMode::Test), (show_ui, clear_canvas))
            .add_systems(OnExit(AppMode::Test), hide_ui)
            .add_systems(
                Update,
                (clear_on_key, update_recognized_text).run_if(in_state(AppMode::Test)),
            );
    }
}

#[derive(Resource, Default)]
struct RecognizedText(String);

#[derive(Resource, Default)]
struct PendingOcr {
    ink: Option<Ink>,
    due: f32,
    primed: bool,
}

/// The single full-window writing surface used by test mode.
#[derive(Component)]
struct TestCanvas;

#[derive(Component)]
struct TestUiRoot;

#[derive(Component)]
struct RecognizedTextLabel;

fn test_mode_hint(checkpoint: Option<&OcrCheckpointSource>) -> String {
    match checkpoint {
        Some(src) => format!("Write in the canvas — model: {}", src.0),
        None => "Write in the canvas — recognized text appears below.".to_string(),
    }
}

fn setup_ui(mut commands: Commands, checkpoint: Option<Res<OcrCheckpointSource>>) {
    // A full-window (minus a little margin) writable canvas, behind the info
    // panel. Plain background so it doesn't compete visually with the ink.
    let mut canvas = commands.spawn((
        TestCanvas,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(0.0),
            left: Val::Px(0.0),
            right: Val::Px(0.0),
            bottom: Val::Px(0.0),
            ..default()
        },
    ));
    make_writable(&mut canvas);

    commands.spawn((
        TestUiRoot,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            padding: UiRect::all(Val::Px(8.0)),
            max_width: Val::Px(700.0),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.4)),
        Pickable::IGNORE,
        GlobalZIndex(10),
        children![
            (
                Text::new(test_mode_hint(checkpoint.as_deref())),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.7)),
            ),
            (
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(28.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                RecognizedTextLabel,
            ),
        ],
    ));
}

fn show_ui(mut roots: Query<&mut Visibility, Or<(With<TestUiRoot>, With<TestCanvas>)>>) {
    for mut vis in &mut roots {
        *vis = Visibility::Inherited;
    }
}

fn hide_ui(mut roots: Query<&mut Visibility, Or<(With<TestUiRoot>, With<TestCanvas>)>>) {
    for mut vis in &mut roots {
        *vis = Visibility::Hidden;
    }
}

fn clear_canvas(
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut pending: ResMut<PendingOcr>,
) {
    pending.ink = None;
    pending.primed = false;
    for mut cell in &mut canvas {
        cell.clear();
    }
}

fn clear_on_key(
    keys: Res<ButtonInput<KeyCode>>,
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut pending: ResMut<PendingOcr>,
) {
    if keys.just_pressed(KeyCode::Space) || keys.just_pressed(KeyCode::Escape) {
        pending.ink = None;
        pending.primed = false;
        for mut cell in &mut canvas {
            cell.clear();
        }
    }
}

fn update_recognized_text(
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut recognized: ResMut<RecognizedText>,
    mut pending: ResMut<PendingOcr>,
    ocr: Option<NonSend<OcrRecognizer>>,
    mut labels: Query<&mut Text, With<RecognizedTextLabel>>,
    time: Res<Time>,
) {
    let Ok(mut cell) = canvas.single_mut() else {
        return;
    };
    let Some(ocr) = ocr else { return };

    if cell.just_finished {
        cell.just_finished = false;
        if ocr.is_vlm() {
            pending.ink = Some(cell.ink.clone());
            pending.due = time.elapsed_secs() + VLM_DEBOUNCE;
            pending.primed = false;
        } else {
            apply_recognition(&ocr, &cell.ink, &mut recognized, &mut labels);
        }
        return;
    }

    if pending.ink.is_none() {
        return;
    }
    if time.elapsed_secs() < pending.due {
        return;
    }
    if !pending.primed {
        // One frame so "recognizing…" can paint before Metal inference blocks.
        pending.primed = true;
        for mut label in &mut labels {
            label.0 = "recognizing…".to_string();
        }
        return;
    }
    let ink = pending.ink.take().expect("pending ink was Some");
    pending.primed = false;
    apply_recognition(&ocr, &ink, &mut recognized, &mut labels);
}

fn apply_recognition(
    ocr: &OcrRecognizer,
    ink: &Ink,
    recognized: &mut RecognizedText,
    labels: &mut Query<&mut Text, With<RecognizedTextLabel>>,
) {
    match ocr.recognize(ink) {
        Ok(text) => recognized.0 = text,
        Err(err) => {
            eprintln!("recognition failed: {err}");
            return;
        }
    }

    for mut label in labels.iter_mut() {
        label.0 = if recognized.0.is_empty() {
            "(empty)".to_string()
        } else {
            recognized.0.clone()
        };
    }
}
