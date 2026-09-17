//! App mode: switch between the calibration flow and the free-write test
//! area, via a small button bar (top-right) always on screen.

use bevy::prelude::*;

use crate::ui_theme::ui_font;
use crate::writing_cell::stop_write_bubbling;

#[derive(States, Clone, Copy, Eq, PartialEq, Hash, Debug, Default)]
pub enum AppMode {
    #[default]
    Test,
    Calibrate,
    Review,
}

pub struct ModePlugin;

impl Plugin for ModePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<AppMode>()
            .add_systems(Startup, setup_mode_bar);
    }
}

fn setup_mode_bar(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                right: Val::Px(12.0),
                column_gap: Val::Px(8.0),
                ..default()
            },
            // The writable canvas/grid in each mode is a separate,
            // unparented full-screen root with no explicit z-index, so
            // without this, picking priority between it and this bar was
            // undefined — clicks on the buttons could be swallowed by the
            // canvas underneath instead. This keeps the mode bar on top.
            GlobalZIndex(100),
        ))
        .with_children(|parent| {
            mode_button(parent, "Test", AppMode::Test);
            mode_button(parent, "Calibrate", AppMode::Calibrate);
            mode_button(parent, "Review", AppMode::Review);
        });
}

fn mode_button(parent: &mut ChildSpawnerCommands, label: &str, mode: AppMode) {
    let mut button = parent.spawn((
        Button,
        Node {
            padding: UiRect::axes(Val::Px(14.0), Val::Px(8.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.12)),
    ));
    button
        .with_children(|b| {
            b.spawn((
                Text::new(label),
                ui_font(18.0),
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        })
        .observe(
            move |_: On<Pointer<Click>>, mut next: ResMut<NextState<AppMode>>| {
                next.set(mode);
            },
        );
    stop_write_bubbling(&mut button);
}
