//! A "writing cell": any UI entity that can be written on with mouse, touch,
//! or pen, via bevy_picking's unified pointer events. Used both for the
//! single big canvas in test mode and for each small prompt cell in
//! calibration mode's grid.
//!
//! Picking dispatches `Press`/`Drag`/`Release` to whichever entity the
//! interaction actually started on (and bubbles them to ancestors), so each
//! cell only ever accumulates strokes drawn inside it — no manual hit-testing
//! against cell bounds needed.

use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use hwr_ink::ink::Ink;

pub struct WritingCellPlugin;

impl Plugin for WritingCellPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw_cell_ink);
    }
}

/// The ink accumulated by one writing cell. `just_finished` is set on pen-up
/// (one stroke completed) — consumers (e.g. test mode's recognizer) should
/// check and clear it.
#[derive(Component, Default)]
pub struct CellInk {
    pub ink: Ink,
    pub just_finished: bool,
    /// Samples accepted from this cell so far. Not touched by `clear()` —
    /// calibration mode uses it to show a running per-prompt count.
    pub saved_count: usize,
    pen_down: bool,
    stroke_start: f32,
}

impl CellInk {
    /// Clear the current stroke(s), keeping `saved_count`.
    pub fn clear(&mut self) {
        self.ink.clear();
        self.pen_down = false;
        self.just_finished = false;
    }
}

/// Make `entity` writable: attach a `CellInk` and the pointer observers that
/// feed it from mouse/touch/pen input.
pub fn make_writable(entity: &mut EntityCommands) {
    entity.insert(CellInk::default());
    entity.observe(on_press);
    entity.observe(on_drag);
    entity.observe(on_release);
}

/// Stop `Press`/`Drag`/`Release` from bubbling past this entity — attach to
/// buttons (or anything else clickable) that sit on top of a writable cell,
/// so clicking them doesn't also draw a stroke on the cell underneath.
pub fn stop_write_bubbling(entity: &mut EntityCommands) {
    entity.observe(|mut trigger: On<Pointer<Press>>| trigger.propagate(false));
    entity.observe(|mut trigger: On<Pointer<Drag>>| trigger.propagate(false));
    entity.observe(|mut trigger: On<Pointer<Release>>| trigger.propagate(false));
}

fn on_press(trigger: On<Pointer<Press>>, mut cells: Query<&mut CellInk>, time: Res<Time>) {
    if trigger.event.button != PointerButton::Primary {
        return;
    }
    let Ok(mut cell) = cells.get_mut(trigger.entity) else {
        return;
    };
    let pos = trigger.pointer_location.position;
    cell.pen_down = true;
    cell.stroke_start = time.elapsed_secs();
    cell.ink.push(pos.x, pos.y, 0.0);
}

fn on_drag(trigger: On<Pointer<Drag>>, mut cells: Query<&mut CellInk>, time: Res<Time>) {
    let Ok(mut cell) = cells.get_mut(trigger.entity) else {
        return;
    };
    if !cell.pen_down {
        return;
    }
    let pos = trigger.pointer_location.position;
    let dt = time.elapsed_secs() - cell.stroke_start;
    cell.ink.push(pos.x, pos.y, dt);
}

fn on_release(trigger: On<Pointer<Release>>, mut cells: Query<&mut CellInk>, time: Res<Time>) {
    if trigger.event.button != PointerButton::Primary {
        return;
    }
    let Ok(mut cell) = cells.get_mut(trigger.entity) else {
        return;
    };
    if !cell.pen_down {
        return;
    }
    let pos = trigger.pointer_location.position;
    let dt = time.elapsed_secs() - cell.stroke_start;
    cell.ink.push(pos.x, pos.y, dt);
    cell.ink.pen_up();
    cell.pen_down = false;
    cell.just_finished = true;
}

const STROKE_COLOR: Color = Color::srgb(0.9, 0.9, 0.95);

/// Draw every cell's ink as connected line segments in screen space,
/// including the in-progress stroke (so lines appear as you write, not only
/// after you lift the pen).
///
/// Ink coordinates are pixel space (y-down, origin top-left, matching
/// `pointer_location.position`); gizmos draw in 2D world space (y-up, origin
/// center), so we flip/offset using the window size each frame.
fn draw_cell_ink(
    cells: Query<(&CellInk, &InheritedVisibility)>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut gizmos: Gizmos,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let half = Vec2::new(window.width(), window.height()) / 2.0;
    let to_world = |x: f32, y: f32| Vec2::new(x - half.x, half.y - y);

    for (cell, visibility) in &cells {
        // Gizmos aren't part of bevy_ui's render tree, so a hidden cell
        // (e.g. another mode's canvas) wouldn't otherwise stop drawing.
        if !visibility.get() {
            continue;
        }
        for stroke in cell.ink.strokes_with_open() {
            for pair in stroke.windows(2) {
                let a = to_world(pair[0].x, pair[0].y);
                let b = to_world(pair[1].x, pair[1].y);
                gizmos.line_2d(a, b, STROKE_COLOR);
            }
        }
    }
}
