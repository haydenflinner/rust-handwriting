//! Test mode: write in the canvas, see what the recognizer decodes.

use bevy::prelude::*;

use crate::mode::AppMode;
use crate::ocr::OcrRecognizer;
use crate::writing_cell::{make_writable, CellInk};

pub struct TestModePlugin;

impl Plugin for TestModePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RecognizedText::default())
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

/// The single full-window writing surface used by test mode.
#[derive(Component)]
struct TestCanvas;

#[derive(Component)]
struct TestUiRoot;

#[derive(Component)]
struct RecognizedTextLabel;

fn setup_ui(mut commands: Commands) {
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
                Text::new("Write in the canvas — recognized text appears below."),
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

fn show_ui(
    mut roots: Query<&mut Visibility, Or<(With<TestUiRoot>, With<TestCanvas>)>>,
) {
    for mut vis in &mut roots {
        *vis = Visibility::Inherited;
    }
}

fn hide_ui(mut roots: Query<&mut Visibility, Or<(With<TestUiRoot>, With<TestCanvas>)>>) {
    for mut vis in &mut roots {
        *vis = Visibility::Hidden;
    }
}

fn clear_canvas(mut canvas: Query<&mut CellInk, With<TestCanvas>>) {
    for mut cell in &mut canvas {
        cell.clear();
    }
}

fn clear_on_key(keys: Res<ButtonInput<KeyCode>>, mut canvas: Query<&mut CellInk, With<TestCanvas>>) {
    if keys.just_pressed(KeyCode::Space) || keys.just_pressed(KeyCode::Escape) {
        for mut cell in &mut canvas {
            cell.clear();
        }
    }
}

fn update_recognized_text(
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut recognized: ResMut<RecognizedText>,
    ocr: Option<NonSend<OcrRecognizer>>,
    mut labels: Query<&mut Text, With<RecognizedTextLabel>>,
) {
    let Ok(mut cell) = canvas.single_mut() else {
        return;
    };
    if !cell.just_finished {
        return;
    }
    cell.just_finished = false;

    let Some(ocr) = ocr else { return };

    match ocr.0.recognize_greedy(&cell.ink) {
        Ok(text) => recognized.0 = text,
        Err(err) => {
            eprintln!("recognition failed: {err}");
            return;
        }
    }

    for mut label in &mut labels {
        label.0 = if recognized.0.is_empty() {
            "(empty)".to_string()
        } else {
            recognized.0.clone()
        };
    }
}
