mod calibrate;
mod mode;
mod ocr;
mod prompts;
mod review;
mod storage;
mod test_mode;
mod writing_cell;

use bevy::prelude::*;

const BACKGROUND: Color = Color::srgb(0.08, 0.08, 0.1);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "hwr".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(BACKGROUND))
        .add_plugins((
            writing_cell::WritingCellPlugin,
            ocr::OcrPlugin,
            mode::ModePlugin,
            test_mode::TestModePlugin,
            calibrate::CalibratePlugin,
            review::ReviewPlugin,
        ))
        .add_systems(Startup, setup_camera)
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
