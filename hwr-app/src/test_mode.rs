//! Test mode: write in the canvas, see what the recognizer decodes.

use std::collections::HashMap;

use bevy::prelude::*;

use hwr_ink::ink::Ink;

use crate::hunyuan_tasks::{self, HunyuanTask, TASKS};
use crate::mode::AppMode;
use crate::ocr::{OcrCheckpointSource, OcrClient, UiPointerDown};
use crate::writing_cell::{make_writable, stop_write_bubbling, CellInk};

/// Pause after the last pen-up before calling a VLM, so several digits or
/// letters written as separate strokes are recognized once.
const VLM_DEBOUNCE: f32 = 1.0;

const MENU_BG: Color = Color::srgba(0.07, 0.07, 0.09, 0.96);
const OPTION_BG: Color = Color::srgba(1.0, 1.0, 1.0, 0.08);
const OPTION_SELECTED_BG: Color = Color::srgba(0.35, 0.55, 0.95, 0.4);

pub struct TestModePlugin;

impl Plugin for TestModePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RecognizedText::default())
            .insert_resource(PendingOcr::default())
            .insert_resource(VlTaskState::default())
            .add_systems(Startup, setup_ui)
            .add_systems(OnEnter(AppMode::Test), (show_ui, clear_canvas))
            .add_systems(OnExit(AppMode::Test), hide_ui)
            .add_systems(
                Update,
                (clear_on_key, update_recognized_text, sync_task_dropdown)
                    .chain()
                    .run_if(in_state(AppMode::Test)),
            );
    }
}

#[derive(Resource, Default)]
struct RecognizedText(String);

#[derive(Resource, Default)]
struct PendingOcr {
    ink: Option<Ink>,
    due: f32,
}

/// Hunyuan task picker + per-ink result cache for the dropdown.
#[derive(Resource)]
struct VlTaskState {
    selected: String,
    menu_open: bool,
    last_ink: Option<Ink>,
    results: HashMap<String, String>,
    /// OCR job id → task id, so a prompt-switch still caches the in-flight job.
    submitted: HashMap<u64, String>,
    rerun: bool,
}

impl Default for VlTaskState {
    fn default() -> Self {
        Self {
            selected: hunyuan_tasks::initial_task_id().to_string(),
            menu_open: false,
            last_ink: None,
            results: HashMap::new(),
            submitted: HashMap::new(),
            rerun: false,
        }
    }
}

impl VlTaskState {
    fn forget_ink(&mut self) {
        self.last_ink = None;
        self.results.clear();
        self.submitted.clear();
        self.rerun = false;
        self.menu_open = false;
    }
}

/// The single full-window writing surface used by test mode.
#[derive(Component)]
struct TestCanvas;

#[derive(Component)]
struct TestUiRoot;

#[derive(Component)]
struct RecognizedTextLabel;

#[derive(Component)]
struct TrialLogLabel;

#[derive(Component)]
struct TaskMenuHeaderLabel;

#[derive(Component)]
struct TaskPromptPreview;

#[derive(Component)]
struct TaskMenuList;

#[derive(Component)]
struct TaskOption(&'static str);

#[derive(Component)]
struct TaskDropdownRoot;

fn test_mode_hint(checkpoint: Option<&OcrCheckpointSource>, vlm: bool) -> String {
    let model = match checkpoint {
        Some(src) => format!("model: {}", src.0),
        None => "recognized text appears below".to_string(),
    };
    if vlm {
        format!("Write, then pick a Hunyuan task to re-run this ink. Space/Esc clears. {model}")
    } else {
        format!("Write in the canvas — {model}")
    }
}

fn setup_ui(
    mut commands: Commands,
    checkpoint: Option<Res<OcrCheckpointSource>>,
    ocr: Option<NonSend<OcrClient>>,
) {
    let vlm = ocr.map(|ocr| ocr.is_vlm()).unwrap_or(false);

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

    commands
        .spawn((
            TestUiRoot,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                padding: UiRect::all(Val::Px(8.0)),
                max_width: Val::Px(720.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.4)),
            Pickable::IGNORE,
            GlobalZIndex(10),
        ))
        .with_children(|root| {
            root.spawn((
                Text::new(test_mode_hint(checkpoint.as_deref(), vlm)),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.7)),
                Pickable::IGNORE,
            ));
            if vlm {
                spawn_task_dropdown(root);
            }
            root.spawn((
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(24.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                RecognizedTextLabel,
                Pickable::IGNORE,
            ));
            root.spawn((
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.55)),
                TrialLogLabel,
                Pickable::IGNORE,
            ));
        });
}

fn spawn_task_dropdown(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            TaskDropdownRoot,
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                align_items: AlignItems::FlexStart,
                ..default()
            },
        ))
        .with_children(|col| {
            let mut header = col.spawn((
                Button,
                Node {
                    padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                    min_width: Val::Px(320.0),
                    ..default()
                },
                BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.14)),
            ));
            header
                .with_children(|b| {
                    b.spawn((
                        Text::new(""),
                        TextFont {
                            font_size: FontSize::Px(15.0),
                            ..default()
                        },
                        TextColor(Color::WHITE),
                        TaskMenuHeaderLabel,
                        Pickable::IGNORE,
                    ));
                })
                .observe(
                    |mut trigger: On<Pointer<Click>>, mut state: ResMut<VlTaskState>| {
                        trigger.propagate(false);
                        state.menu_open = !state.menu_open;
                    },
                );
            stop_write_bubbling(&mut header);

            col.spawn((
                TaskMenuList,
                Visibility::Hidden,
                Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(2.0),
                    padding: UiRect::all(Val::Px(4.0)),
                    min_width: Val::Px(360.0),
                    ..default()
                },
                BackgroundColor(MENU_BG),
                GlobalZIndex(110),
            ))
            .with_children(|list| {
                for task in TASKS {
                    spawn_task_option(list, task);
                }
            });

            col.spawn((
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.5)),
                TaskPromptPreview,
                Pickable::IGNORE,
            ));
        });
}

fn spawn_task_option(parent: &mut ChildSpawnerCommands, task: &'static HunyuanTask) {
    let id = task.id;
    let mut button = parent.spawn((
        Button,
        TaskOption(id),
        Node {
            padding: UiRect::axes(Val::Px(8.0), Val::Px(5.0)),
            width: Val::Percent(100.0),
            ..default()
        },
        BackgroundColor(OPTION_BG),
    ));
    button
        .with_children(|b| {
            b.spawn((
                Text::new(format!("{id}  —  {}", task.blurb)),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(Color::WHITE),
                Pickable::IGNORE,
            ));
        })
        .observe(
            move |mut trigger: On<Pointer<Click>>, mut state: ResMut<VlTaskState>| {
                trigger.propagate(false);
                if state.selected != id {
                    state.selected = id.to_string();
                    state.rerun = true;
                }
                state.menu_open = false;
            },
        );
    stop_write_bubbling(&mut button);
}

fn show_ui(mut roots: Query<&mut Visibility, Or<(With<TestUiRoot>, With<TestCanvas>)>>) {
    for mut vis in &mut roots {
        *vis = Visibility::Inherited;
    }
}

fn hide_ui(
    mut roots: Query<&mut Visibility, Or<(With<TestUiRoot>, With<TestCanvas>)>>,
    mut state: ResMut<VlTaskState>,
) {
    state.menu_open = false;
    for mut vis in &mut roots {
        *vis = Visibility::Hidden;
    }
}

fn clear_canvas(
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut pending: ResMut<PendingOcr>,
    mut state: ResMut<VlTaskState>,
    ocr: Option<NonSendMut<OcrClient>>,
    writing: Res<UiPointerDown>,
) {
    pending.ink = None;
    state.forget_ink();
    writing.set(false);
    if let Some(mut ocr) = ocr {
        ocr.cancel();
    }
    for mut cell in &mut canvas {
        cell.clear();
    }
}

fn clear_on_key(
    keys: Res<ButtonInput<KeyCode>>,
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut pending: ResMut<PendingOcr>,
    mut state: ResMut<VlTaskState>,
    mut recognized: ResMut<RecognizedText>,
    mut labels: Query<&mut Text, With<RecognizedTextLabel>>,
    mut logs: Query<&mut Text, (With<TrialLogLabel>, Without<RecognizedTextLabel>)>,
    ocr: Option<NonSendMut<OcrClient>>,
    writing: Res<UiPointerDown>,
) {
    if keys.just_pressed(KeyCode::Space) || keys.just_pressed(KeyCode::Escape) {
        pending.ink = None;
        state.forget_ink();
        recognized.0.clear();
        writing.set(false);
        if let Some(mut ocr) = ocr {
            ocr.cancel();
        }
        for mut cell in &mut canvas {
            cell.clear();
        }
        for mut label in &mut labels {
            label.0.clear();
        }
        for mut log in &mut logs {
            log.0.clear();
        }
    }
}

fn update_recognized_text(
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut recognized: ResMut<RecognizedText>,
    mut pending: ResMut<PendingOcr>,
    mut state: ResMut<VlTaskState>,
    ocr: Option<NonSendMut<OcrClient>>,
    mut labels: Query<&mut Text, With<RecognizedTextLabel>>,
    mut logs: Query<&mut Text, (With<TrialLogLabel>, Without<RecognizedTextLabel>)>,
    time: Res<Time>,
) {
    let Ok(mut cell) = canvas.single_mut() else {
        return;
    };
    let Some(mut ocr) = ocr else { return };

    while let Some((id, current, result)) = ocr.poll() {
        if ocr.is_vlm() {
            if let Some(task) = state.submitted.remove(&id) {
                apply_vlm_result(
                    result,
                    &task,
                    &mut state,
                    &mut recognized,
                    &mut labels,
                    &mut logs,
                );
            }
        } else if current {
            apply_result(result, &mut recognized, &mut labels);
        }
    }

    if ocr.is_vlm() && cell.is_writing() {
        // A new stroke started: drop any in-flight job so Metal is free, and
        // wait until 1s after this stroke (and any that follow) before trying
        // again.
        ocr.cancel();
        pending.ink = None;
        state.forget_ink();
        return;
    }

    if cell.just_finished {
        cell.just_finished = false;
        if ocr.is_vlm() {
            pending.ink = Some(cell.ink.clone());
            pending.due = time.elapsed_secs() + VLM_DEBOUNCE;
        } else {
            pending.ink = None;
            submit_hat(&mut ocr, cell.ink.clone(), &mut labels);
        }
    }

    if ocr.is_vlm() && state.rerun {
        state.rerun = false;
        if pending.ink.is_none() {
            if state.results.contains_key(&state.selected) {
                show_cached(&state, &mut recognized, &mut labels, &mut logs);
            } else if let Some(ink) = state.last_ink.clone() {
                submit_vlm(&mut ocr, ink, &mut state, &mut labels, &mut logs);
            }
        }
    }

    if pending.ink.is_some() && !cell.is_writing() && time.elapsed_secs() >= pending.due {
        let ink = pending.ink.take().expect("pending ink was Some");
        if ocr.is_vlm() {
            state.last_ink = Some(ink.clone());
            state.results.clear();
            submit_vlm(&mut ocr, ink, &mut state, &mut labels, &mut logs);
        } else {
            submit_hat(&mut ocr, ink, &mut labels);
        }
    }
}

fn submit_hat(
    ocr: &mut OcrClient,
    ink: Ink,
    labels: &mut Query<&mut Text, With<RecognizedTextLabel>>,
) {
    ocr.submit(ink, None);
    for mut label in labels.iter_mut() {
        label.0 = "recognizing…".to_string();
    }
}

fn submit_vlm(
    ocr: &mut OcrClient,
    ink: Ink,
    state: &mut VlTaskState,
    labels: &mut Query<&mut Text, With<RecognizedTextLabel>>,
    logs: &mut Query<&mut Text, (With<TrialLogLabel>, Without<RecognizedTextLabel>)>,
) {
    let prompt = hunyuan_tasks::prompt_for(&state.selected)
        .unwrap_or_else(|| hunyuan_tasks::prompt_for(hunyuan_tasks::DEFAULT_TASK_ID).unwrap())
        .to_string();
    let task = state.selected.clone();
    let id = ocr.submit(ink, Some(prompt));
    state.submitted.insert(id, task.clone());
    for mut label in labels.iter_mut() {
        label.0 = format!("recognizing {task}…");
    }
    for mut log in logs.iter_mut() {
        log.0 = format_trial_log(state);
    }
}

fn apply_vlm_result(
    result: Result<String, String>,
    task: &str,
    state: &mut VlTaskState,
    recognized: &mut RecognizedText,
    labels: &mut Query<&mut Text, With<RecognizedTextLabel>>,
    logs: &mut Query<&mut Text, (With<TrialLogLabel>, Without<RecognizedTextLabel>)>,
) {
    match result {
        Ok(text) => {
            state.results.insert(task.to_string(), text);
        }
        Err(err) => {
            eprintln!("recognition failed ({task}): {err}");
            state
                .results
                .insert(task.to_string(), format!("(error: {err})"));
        }
    }
    if task == state.selected {
        show_cached(state, recognized, labels, logs);
    } else {
        for mut log in logs.iter_mut() {
            log.0 = format_trial_log(state);
        }
    }
}

fn show_cached(
    state: &VlTaskState,
    recognized: &mut RecognizedText,
    labels: &mut Query<&mut Text, With<RecognizedTextLabel>>,
    logs: &mut Query<&mut Text, (With<TrialLogLabel>, Without<RecognizedTextLabel>)>,
) {
    recognized.0 = state
        .results
        .get(&state.selected)
        .cloned()
        .unwrap_or_default();
    for mut label in labels.iter_mut() {
        label.0 = if recognized.0.is_empty() {
            "(empty)".to_string()
        } else {
            recognized.0.clone()
        };
    }
    for mut log in logs.iter_mut() {
        log.0 = format_trial_log(state);
    }
}

fn format_trial_log(state: &VlTaskState) -> String {
    if state.results.is_empty() {
        return String::new();
    }
    let mut lines = Vec::new();
    for task in TASKS {
        let Some(text) = state.results.get(task.id) else {
            continue;
        };
        let mark = if task.id == state.selected {
            "▸"
        } else {
            " "
        };
        lines.push(format!("{mark} {}: {}", task.id, clip(text, 240)));
    }
    if lines.len() > 1 {
        lines.join("\n")
    } else {
        String::new()
    }
}

fn clip(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        text.to_string()
    } else {
        format!("{}…", text.chars().take(max_chars).collect::<String>())
    }
}

fn apply_result(
    result: Result<String, String>,
    recognized: &mut RecognizedText,
    labels: &mut Query<&mut Text, With<RecognizedTextLabel>>,
) {
    match result {
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

fn sync_task_dropdown(
    state: Res<VlTaskState>,
    mut header: Query<&mut Text, With<TaskMenuHeaderLabel>>,
    mut preview: Query<&mut Text, (With<TaskPromptPreview>, Without<TaskMenuHeaderLabel>)>,
    mut list: Query<&mut Visibility, With<TaskMenuList>>,
    mut options: Query<(&TaskOption, &mut BackgroundColor)>,
) {
    let task = hunyuan_tasks::task_by_id(&state.selected);
    let arrow = if state.menu_open { "▴" } else { "▾" };
    let header_text = match task {
        Some(task) => format!("{arrow}  {}  —  {}", task.id, task.blurb),
        None => format!("{arrow}  {}", state.selected),
    };
    for mut text in &mut header {
        text.0 = header_text.clone();
    }
    let prompt = task.map(|task| task.prompt).unwrap_or("");
    for mut text in &mut preview {
        text.0 = prompt.to_string();
    }
    for mut vis in &mut list {
        *vis = if state.menu_open {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
    for (option, mut bg) in &mut options {
        *bg = BackgroundColor(if option.0 == state.selected {
            OPTION_SELECTED_BG
        } else {
            OPTION_BG
        });
    }
}
