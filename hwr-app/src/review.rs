//! Review mode: page through the corpus the recognizer is trained/
//! calibrated on — your own `calibration.txt` plus armrest's bundled
//! `data/inks/*.txt` files — rendering each entry's *raw saved ink* next to
//! its label.
//!
//! This exists to catch data-loading bugs (flipped/rotated coordinates,
//! mis-parsed fields, truncated strokes...) by eye: if something's wrong
//! with how ink gets loaded from disk, a page of it will visibly not look
//! like handwriting anymore, which is a lot faster to notice than staring at
//! rows of floats.

use std::path::Path;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use hwr_ink::ink::Ink;
use hwr_model::corpus;

use crate::mode::AppMode;
use crate::storage::calibration_file_path;

const ENTRIES_PER_PAGE: usize = 40;
const INK_COLOR: Color = Color::srgb(0.4, 0.9, 0.6);

pub struct ReviewPlugin;

impl Plugin for ReviewPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ReviewSession::new())
            .add_systems(Startup, setup_ui)
            .add_systems(OnEnter(AppMode::Review), show_ui)
            .add_systems(OnExit(AppMode::Review), hide_ui)
            .add_systems(
                Update,
                (rebuild_grid_on_page_change, draw_review_cells).chain(),
            );
    }
}

struct ReviewPageData {
    title: String,
    entries: Vec<(String, Ink)>,
}

#[derive(Resource)]
struct ReviewSession {
    pages: Vec<ReviewPageData>,
    current: usize,
}

impl ReviewSession {
    fn new() -> Self {
        ReviewSession {
            pages: build_pages(),
            current: 0,
        }
    }
}

/// Load every source we know about and chunk each into fixed-size pages.
/// Missing/empty sources are skipped, not errors — a fresh install has no
/// calibration data yet, and the armrest corpus is only there when run from
/// the workspace root during development.
fn build_pages() -> Vec<ReviewPageData> {
    let mut sources: Vec<(String, Vec<(String, Ink)>)> = Vec::new();

    match corpus::load_pairs(calibration_file_path()) {
        Ok(pairs) if !pairs.is_empty() => sources.push(("My calibration data".to_string(), pairs)),
        _ => {}
    }

    let armrest_inks = Path::new("armrest/data/inks");
    for name in [
        "jabberwocky.txt",
        "prufrock.txt",
        "if-commands.txt",
        "if-transcript.txt",
    ] {
        let path = armrest_inks.join(name);
        match corpus::load_pairs(&path) {
            Ok(pairs) if !pairs.is_empty() => sources.push((name.to_string(), pairs)),
            Ok(_) => {}
            Err(err) => eprintln!(
                "review: couldn't load {} ({err}) — run `cargo run -p hwr-app` from the \
                 workspace root to see armrest's bundled corpus here",
                path.display()
            ),
        }
    }

    let mut pages = Vec::new();
    for (name, entries) in sources {
        let chunks: Vec<Vec<(String, Ink)>> = entries
            .chunks(ENTRIES_PER_PAGE)
            .map(|c| c.to_vec())
            .collect();
        let total = chunks.len();
        for (i, chunk) in chunks.into_iter().enumerate() {
            let title = if total > 1 {
                format!("{name} ({}/{total})", i + 1)
            } else {
                name.clone()
            };
            pages.push(ReviewPageData {
                title,
                entries: chunk,
            });
        }
    }
    pages
}

#[derive(Component)]
struct ReviewUiRoot;

#[derive(Component)]
struct ReviewTitleLabel;

#[derive(Component)]
struct ReviewProgressLabel;

#[derive(Component)]
struct ReviewGridContainer;

#[derive(Component)]
struct ReviewCell(Ink);

fn setup_ui(mut commands: Commands) {
    commands
        .spawn((
            ReviewUiRoot,
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
                            font_size: FontSize::Px(22.0),
                            ..default()
                        },
                        TextColor(Color::WHITE),
                        ReviewTitleLabel,
                    ));
                    nav_button(row, "Next >", 1);
                    row.spawn((
                        Text::new(""),
                        TextFont {
                            font_size: FontSize::Px(14.0),
                            ..default()
                        },
                        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.6)),
                        ReviewProgressLabel,
                    ));
                });
            parent.spawn((
                Text::new(
                    "Read-only: this is the raw ink actually loaded from disk for each source, \
                     not a re-drawing — if something's wrong with how it's parsed or \
                     normalized, it'll look wrong here.",
                ),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.55)),
            ));
            parent.spawn((
                ReviewGridContainer,
                Node {
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: Val::Px(8.0),
                    row_gap: Val::Px(8.0),
                    align_content: AlignContent::FlexStart,
                    flex_grow: 1.0,
                    ..default()
                },
            ));
        });
}

fn nav_button(parent: &mut ChildSpawnerCommands, label: &str, delta: i32) {
    parent
        .spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.12)),
        ))
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
            move |mut trigger: On<Pointer<Click>>, mut session: ResMut<ReviewSession>| {
                trigger.propagate(false);
                let len = session.pages.len() as i32;
                if len == 0 {
                    return;
                }
                session.current = (session.current as i32 + delta).rem_euclid(len) as usize;
            },
        );
    // No writable canvas sits under review mode's own UI, but the mode bar
    // does — keep buttons well-behaved regardless.
}

fn rebuild_grid_on_page_change(
    session: Res<ReviewSession>,
    mut commands: Commands,
    grid: Query<Entity, With<ReviewGridContainer>>,
    mut title_labels: Query<&mut Text, (With<ReviewTitleLabel>, Without<ReviewProgressLabel>)>,
    mut progress_labels: Query<&mut Text, (With<ReviewProgressLabel>, Without<ReviewTitleLabel>)>,
) {
    if !session.is_changed() {
        return;
    }
    let Ok(grid_entity) = grid.single() else {
        return;
    };

    if session.pages.is_empty() {
        for mut text in &mut title_labels {
            text.0 = "No data found".to_string();
        }
        for mut text in &mut progress_labels {
            text.0 =
                "No calibration.txt yet, and armrest/data/inks/ isn't visible from here."
                    .to_string();
        }
        commands.entity(grid_entity).despawn_children();
        return;
    }

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
    let entries = page.entries.clone();
    commands.entity(grid_entity).with_children(|parent| {
        for (text, ink) in entries {
            spawn_review_cell(parent, text, ink);
        }
    });
}

fn spawn_review_cell(parent: &mut ChildSpawnerCommands, text: String, ink: Ink) {
    parent
        .spawn((
            ReviewCell(ink),
            Node {
                width: Val::Px(150.0),
                height: Val::Px(110.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(4.0)),
                border: UiRect::all(Val::Px(1.0)),
                overflow: Overflow::clip(),
                ..default()
            },
            BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.04)),
            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.15)),
            Pickable::IGNORE,
        ))
        .with_children(|c| {
            c.spawn((
                Text::new(truncate(&text, 24)),
                TextFont {
                    font_size: FontSize::Px(11.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
                Pickable::IGNORE,
            ));
        });
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max_chars).collect();
        t.push('…');
        t
    }
}

/// Scale + center `ink` (in whatever coordinate space it was recorded in —
/// armrest's bundled files and our own calibration data use different
/// scales) to fit inside `target_size`, centered on `target_center`, all in
/// the same window-pixel space as `UiGlobalTransform`/`pointer_location`.
fn fit_transform(ink: &Ink, target_center: Vec2, target_size: Vec2) -> impl Fn(f32, f32) -> Vec2 {
    let w = (ink.x_range.max - ink.x_range.min).max(1e-3);
    let h = (ink.y_range.max - ink.y_range.min).max(1e-3);
    let scale = (target_size.x / w).min(target_size.y / h);
    let cx = (ink.x_range.min + ink.x_range.max) / 2.0;
    let cy = (ink.y_range.min + ink.y_range.max) / 2.0;
    move |x: f32, y: f32| target_center + Vec2::new((x - cx) * scale, (y - cy) * scale)
}

fn draw_review_cells(
    cells: Query<(&ReviewCell, &ComputedNode, &UiGlobalTransform, &InheritedVisibility)>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut gizmos: Gizmos,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let half = Vec2::new(window.width(), window.height()) / 2.0;
    let window_to_world = |p: Vec2| Vec2::new(p.x - half.x, half.y - p.y);

    // ComputedNode/UiGlobalTransform are in physical pixels; everything else
    // here (window.width/height, pointer_location, and so `to_world` above)
    // is logical pixels. Without this, on any display with a scale factor
    // != 1 the fitted ink ends up positioned/sized wrong by that factor.
    let scale = window.scale_factor();

    for (cell, node, transform, visibility) in &cells {
        // Gizmos aren't part of bevy_ui's render tree, so a hidden page
        // (e.g. while a different mode is active) wouldn't otherwise stop
        // drawing.
        if !visibility.get() {
            continue;
        }
        let ink = &cell.0;
        if ink.is_empty() {
            continue;
        }
        // Node-space origin is the node's center; leave a margin and room
        // at the top for the label.
        let center = transform.transform_point2(Vec2::ZERO) / scale + Vec2::new(0.0, 10.0);
        let target_size = (node.size / scale - Vec2::new(12.0, 30.0)).max(Vec2::splat(1.0));
        let fit = fit_transform(ink, center, target_size);

        for stroke in ink.strokes() {
            for pair in stroke.windows(2) {
                let a = window_to_world(fit(pair[0].x, pair[0].y));
                let b = window_to_world(fit(pair[1].x, pair[1].y));
                gizmos.line_2d(a, b, INK_COLOR);
            }
        }
    }
}

fn show_ui(mut roots: Query<&mut Visibility, With<ReviewUiRoot>>) {
    for mut vis in &mut roots {
        *vis = Visibility::Inherited;
    }
}

fn hide_ui(mut roots: Query<&mut Visibility, With<ReviewUiRoot>>) {
    for mut vis in &mut roots {
        *vis = Visibility::Hidden;
    }
}
