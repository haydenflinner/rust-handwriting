//! Calibration mode: one screen per prompt category (digits, numbers,
//! symbols, words), each a grid of small writable cells. Write as many
//! prompts as you like, then click "Save Page" once to keep every
//! non-empty cell's attempt (each clears for another try); "Clear" on an
//! individual cell discards just that one. "Prev"/"Next" move between
//! category screens.
//!
//! Every accepted sample is appended immediately to a personal training
//! corpus on disk, in armrest's `text\tink` format — directly loadable by
//! `hwr_model::corpus::load_pairs` (and so by `hwr-model`'s `train` binary)
//! for a future fine-tuning pass.

use bevy::prelude::*;

use crate::mode::AppMode;
use crate::prompts::{pages, Page};
use crate::storage;
use crate::writing_cell::{make_writable, stop_write_bubbling, CellInk};

pub struct CalibratePlugin;

impl Plugin for CalibratePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CalibrationSession::new())
            .add_systems(Startup, setup_ui)
            .add_systems(OnEnter(AppMode::Calibrate), show_ui)
            .add_systems(OnExit(AppMode::Calibrate), hide_ui)
            .add_systems(Update, rebuild_grid_on_page_change);
    }
}

#[derive(Resource)]
struct CalibrationSession {
    pages: Vec<Page>,
    current: usize,
}

impl CalibrationSession {
    fn new() -> Self {
        CalibrationSession {
            pages: pages(),
            current: 0,
        }
    }
}

#[derive(Component)]
struct CalibrateUiRoot;

#[derive(Component)]
struct PageTitleLabel;

#[derive(Component)]
struct PageProgressLabel;

#[derive(Component)]
struct GridContainer;

#[derive(Component)]
struct CountLabel;

/// Holds the prompt text for a calibration grid cell, so the page-level
/// "Save Page" button can find it without a closure capturing per-cell state.
#[derive(Component)]
struct CalibrationCell(String);

fn setup_ui(mut commands: Commands) {
    commands
        .spawn((
            CalibrateUiRoot,
            Visibility::Hidden,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                bottom: Val::Px(0.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(12.0)),
                row_gap: Val::Px(8.0),
                ..default()
            },
            GlobalZIndex(10),
        ))
        .with_children(|parent| {
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    nav_button(row, "< Prev", -1);
                    row.spawn((
                        Text::new(""),
                        TextFont {
                            font_size: FontSize::Px(26.0),
                            ..default()
                        },
                        TextColor(Color::WHITE),
                        PageTitleLabel,
                    ));
                    nav_button(row, "Next >", 1);
                    save_page_button(row);
                    row.spawn((
                        Text::new(""),
                        TextFont {
                            font_size: FontSize::Px(14.0),
                            ..default()
                        },
                        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.6)),
                        PageProgressLabel,
                    ));
                });
            parent.spawn((
                Text::new(
                    "Write each prompt in its box, as many times as you like. \
                     Save Page keeps every non-empty box (and clears them for another round); \
                     Clear on a box discards just that one.",
                ),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.55)),
            ));
            parent.spawn((
                GridContainer,
                Node {
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: Val::Px(10.0),
                    row_gap: Val::Px(10.0),
                    align_content: AlignContent::FlexStart,
                    flex_grow: 1.0,
                    ..default()
                },
            ));
        });
}

fn nav_button(parent: &mut ChildSpawnerCommands, label: &str, delta: i32) {
    let mut button = parent.spawn((
        Button,
        Node {
            padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.12)),
    ));
    button
        .with_children(|b| {
            b.spawn((
                Text::new(label),
                TextFont {
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        })
        .observe(
            move |mut trigger: On<Pointer<Click>>, mut session: ResMut<CalibrationSession>| {
                trigger.propagate(false);
                let len = session.pages.len() as i32;
                if len == 0 {
                    return;
                }
                session.current = (session.current as i32 + delta).rem_euclid(len) as usize;
            },
        );
    stop_write_bubbling(&mut button);
}

/// Saves every non-empty cell on the current page in one click: appends
/// `(prompt, ink)` to the log, bumps that cell's counter, and clears it.
fn save_page_button(parent: &mut ChildSpawnerCommands) {
    let mut button = parent.spawn((
        Button,
        Node {
            padding: UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.3, 0.7, 0.4, 0.5)),
    ));
    button
        .with_children(|b| {
            b.spawn((
                Text::new("Save Page"),
                TextFont {
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        })
        .observe(
            move |mut trigger: On<Pointer<Click>>,
                  mut cells: Query<(&mut CellInk, &CalibrationCell, &Children)>,
                  children_q: Query<&Children>,
                  mut count_texts: Query<&mut Text, With<CountLabel>>| {
                trigger.propagate(false);
                for (mut cell, prompt, children) in &mut cells {
                    if !cell.ink.is_empty() {
                        storage::append_calibration_sample(&prompt.0, &cell.ink);
                        cell.saved_count += 1;
                        update_count_label(
                            children,
                            &children_q,
                            &mut count_texts,
                            cell.saved_count,
                        );
                    }
                    cell.clear();
                }
            },
        );
    stop_write_bubbling(&mut button);
}

fn rebuild_grid_on_page_change(
    session: Res<CalibrationSession>,
    mut commands: Commands,
    grid: Query<Entity, With<GridContainer>>,
    mut title_labels: Query<&mut Text, (With<PageTitleLabel>, Without<PageProgressLabel>)>,
    mut progress_labels: Query<&mut Text, (With<PageProgressLabel>, Without<PageTitleLabel>)>,
) {
    if !session.is_changed() {
        return;
    }
    let Ok(grid_entity) = grid.single() else {
        return;
    };
    let Some(page) = session.pages.get(session.current) else {
        return;
    };

    for mut text in &mut title_labels {
        text.0 = page.title.clone();
    }
    for mut text in &mut progress_labels {
        text.0 = format!("Page {} / {}", session.current + 1, session.pages.len());
    }

    commands.entity(grid_entity).despawn_children();
    let prompts = page.prompts.clone();
    commands.entity(grid_entity).with_children(|parent| {
        for prompt in prompts {
            spawn_cell(parent, prompt);
        }
    });
}

fn spawn_cell(parent: &mut ChildSpawnerCommands, prompt: String) {
    let mut cell = parent.spawn((
        CalibrationCell(prompt.clone()),
        Node {
            width: Val::Px(130.0),
            height: Val::Px(120.0),
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::SpaceBetween,
            padding: UiRect::all(Val::Px(6.0)),
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.05)),
        BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.2)),
    ));
    make_writable(&mut cell);
    let cell_entity = cell.id();

    cell.with_children(|c| {
        c.spawn(Node {
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            ..default()
        })
        .with_children(|header| {
            header.spawn((
                Text::new(prompt),
                TextFont {
                    font_size: FontSize::Px(18.0),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
            header.spawn((
                Text::new("×0"),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.5)),
                CountLabel,
            ));
        });

        clear_button(c, cell_entity);
    });
}

fn clear_button(parent: &mut ChildSpawnerCommands, cell_entity: Entity) {
    let mut button = parent.spawn((
        Button,
        Node {
            align_self: AlignSelf::FlexStart,
            padding: UiRect::axes(Val::Px(8.0), Val::Px(3.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.12)),
    ));
    button
        .with_children(|b| {
            b.spawn((
                Text::new("Clear"),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        })
        .observe(
            move |mut trigger: On<Pointer<Click>>, mut cells: Query<&mut CellInk>| {
                trigger.propagate(false);
                if let Ok(mut cell) = cells.get_mut(cell_entity) {
                    cell.clear();
                }
            },
        );
    stop_write_bubbling(&mut button);
}

/// `CountLabel` is a grandchild of the cell (cell -> header row -> label), so
/// walk down one more level looking for it.
fn update_count_label(
    children: &Children,
    children_q: &Query<&Children>,
    count_texts: &mut Query<&mut Text, With<CountLabel>>,
    count: usize,
) {
    for &child in children {
        if let Ok(mut text) = count_texts.get_mut(child) {
            text.0 = format!("×{count}");
            return;
        }
        if let Ok(grandchildren) = children_q.get(child) {
            update_count_label(grandchildren, children_q, count_texts, count);
        }
    }
}

fn show_ui(mut roots: Query<&mut Visibility, With<CalibrateUiRoot>>) {
    for mut vis in &mut roots {
        *vis = Visibility::Inherited;
    }
}

fn hide_ui(mut roots: Query<&mut Visibility, With<CalibrateUiRoot>>) {
    for mut vis in &mut roots {
        *vis = Visibility::Hidden;
    }
}
