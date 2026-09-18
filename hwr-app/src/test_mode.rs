//! Test mode: write in the canvas, see what the recognizer decodes.

use std::collections::HashMap;

use bevy::asset::RenderAssetUsages;
use bevy::clipboard::Clipboard;
use bevy::image::Image;
use bevy::input::mouse::MouseScrollUnit;
use bevy::input_focus::InputFocus;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::text::{EditableText, LineBreak, TextCursorStyle};
use bevy::ui::widget::TextScroll;
use bevy::window::PrimaryWindow;

use hwr_ink::ink::Ink;

use crate::hunyuan_tasks::{self, HunyuanTask, TASKS};
use crate::mode::AppMode;
use crate::ocr::{OcrCheckpointSource, OcrClient, UiPointerDown};
use crate::typst_convert::latex_to_typst;
use crate::typst_preview::{PreviewResult, TypstPreviewClient};
use crate::ui_theme::{result_font, ui_font, ui_font_semibold};
use crate::writing_cell::{make_writable, stop_write_bubbling, CellInk};

/// Pause after the last pen-up before calling a VLM, so several digits or
/// letters written as separate strokes are recognized once.
const VLM_DEBOUNCE: f32 = 1.0;

const MENU_BG: Color = Color::srgba(0.07, 0.07, 0.09, 0.96);
const OPTION_BG: Color = Color::srgba(1.0, 1.0, 1.0, 0.08);
const OPTION_SELECTED_BG: Color = Color::srgba(0.35, 0.55, 0.95, 0.4);
const CARD_BG: Color = Color::srgba(1.0, 1.0, 1.0, 0.06);
const CARD_SELECTED_BG: Color = Color::srgba(0.35, 0.55, 0.95, 0.22);
const PANEL_BG: Color = Color::srgba(0.06, 0.06, 0.09, 0.94);
const PANEL_BORDER: Color = Color::srgba(1.0, 1.0, 1.0, 0.12);
const OUTPUT_CARD_BG: Color = Color::srgba(1.0, 1.0, 1.0, 0.05);
const FIELD_BG: Color = Color::srgba(0.02, 0.02, 0.04, 0.72);
const COPY_BG: Color = Color::srgba(1.0, 1.0, 1.0, 0.12);
const COPY_FLASH_BG: Color = Color::srgba(0.35, 0.72, 0.48, 0.45);
const LATEX_ACCENT: Color = Color::srgb(0.45, 0.72, 0.98);
const TYPST_ACCENT: Color = Color::srgb(0.96, 0.72, 0.38);
const PREVIEW_ACCENT: Color = Color::srgb(0.58, 0.8, 0.58);
const PREVIEW_PAPER: Color = Color::srgb(0.957, 0.945, 0.918);
const COPY_FLASH_SECS: f32 = 1.25;

pub struct TestModePlugin;

impl Plugin for TestModePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(RecognizedText::default())
            .insert_resource(TypstText::default())
            .insert_resource(PendingOcr::default())
            .insert_resource(VlTaskState::default())
            .add_systems(Startup, (setup_preview_client, setup_ui).chain())
            .add_systems(OnEnter(AppMode::Test), (show_ui, clear_canvas))
            .add_systems(OnExit(AppMode::Test), hide_ui)
            .add_systems(
                Update,
                (
                    unfocus_when_writing,
                    clear_on_key,
                    update_recognized_text,
                    sync_task_dropdown,
                    sync_result_ui,
                    sync_typst_preview,
                    fit_typst_preview_image,
                    sync_copy_buttons,
                )
                    .chain()
                    .run_if(in_state(AppMode::Test)),
            );
    }
}

#[derive(Resource, Default)]
struct RecognizedText(String);

#[derive(Resource, Default)]
struct TypstText(String);

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputKind {
    Latex,
    Typst,
}

impl OutputKind {
    fn title(self) -> &'static str {
        match self {
            Self::Latex => "LaTeX",
            Self::Typst => "Typst",
        }
    }

    fn accent(self) -> Color {
        match self {
            Self::Latex => LATEX_ACCENT,
            Self::Typst => TYPST_ACCENT,
        }
    }
}

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
    status: String,
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
            status: String::new(),
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
        self.status.clear();
    }
}

/// The single writing surface used by test mode (right half of the window).
#[derive(Component)]
struct TestCanvas;

#[derive(Component)]
struct TestSplitRoot;

#[derive(Component)]
struct TestUiRoot;

#[derive(Component)]
struct OutputField(OutputKind);

#[derive(Resource)]
struct PreviewTexture(Handle<Image>);

#[derive(Component)]
struct TypstPreviewPaper;

#[derive(Component)]
struct TypstPreviewImage;

#[derive(Component)]
struct TypstPreviewStatus;

#[derive(Component)]
struct TypstPreviewStatusText;

#[derive(Component)]
struct CopyButton {
    kind: OutputKind,
    copied_until: f32,
}

#[derive(Component)]
struct CopyButtonLabel;

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

#[derive(Component)]
struct ResultsScroll;

#[derive(Component)]
struct TaskResultCard(&'static str);

#[derive(Component)]
struct TaskResultBody(&'static str);

fn test_mode_hint(checkpoint: Option<&OcrCheckpointSource>, vlm: bool) -> String {
    let model = match checkpoint {
        Some(src) => format!("model: {}", src.0),
        None => "recognized text appears below".to_string(),
    };
    if vlm {
        format!(
            "Write, then pick a Hunyuan task to re-run this ink. Select output to copy, or use Copy. Scribble over a letter to erase it. Space/Esc clears. {model}"
        )
    } else {
        format!(
            "Write in the canvas. Select output to copy, or use Copy. Scribble over a letter to erase it. Space/Esc clears. {model}"
        )
    }
}

fn setup_preview_client(world: &mut World) {
    world.insert_non_send(TypstPreviewClient::new());
}

fn setup_ui(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    checkpoint: Option<Res<OcrCheckpointSource>>,
    ocr: Option<NonSend<OcrClient>>,
) {
    let vlm = ocr.map(|ocr| ocr.is_vlm()).unwrap_or(false);
    let preview = images.add(placeholder_image());
    commands.insert_resource(PreviewTexture(preview.clone()));

    commands
        .spawn((
            TestSplitRoot,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                right: Val::Px(0.0),
                bottom: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Row,
                ..default()
            },
        ))
        .with_children(|split| {
            split
                .spawn((
                    TestUiRoot,
                    Node {
                        width: Val::Percent(50.0),
                        height: Val::Percent(100.0),
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(10.0),
                        padding: UiRect::all(Val::Px(14.0)),
                        border: UiRect {
                            right: Val::Px(1.0),
                            ..default()
                        },
                        ..default()
                    },
                    BackgroundColor(PANEL_BG),
                    BorderColor {
                        right: PANEL_BORDER,
                        ..default()
                    },
                    GlobalZIndex(10),
                ))
                .with_children(|root| {
                    root.spawn((
                        Text::new(test_mode_hint(checkpoint.as_deref(), vlm)),
                        ui_font(13.0),
                        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.7)),
                        Pickable::IGNORE,
                    ));
                    if vlm {
                        spawn_task_dropdown(root);
                    }
                    spawn_preview_card(root, preview);
                    spawn_output_card(root, OutputKind::Latex);
                    spawn_output_card(root, OutputKind::Typst);
                    // The per-task result strip at the bottom is redundant with
                    // the LaTeX/Typst fields and steals their space.
                    // if vlm {
                    //     spawn_results_scroll(root);
                    // }
                });

            let mut canvas = split.spawn((
                TestCanvas,
                Node {
                    width: Val::Percent(50.0),
                    height: Val::Percent(100.0),
                    ..default()
                },
            ));
            make_writable(&mut canvas);
        });
}

fn spawn_preview_card(parent: &mut ChildSpawnerCommands, image: Handle<Image>) {
    parent
        .spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                width: Val::Percent(100.0),
                flex_grow: 1.2,
                flex_shrink: 1.0,
                min_height: Val::Px(120.0),
                overflow: Overflow::clip(),
                padding: UiRect::all(Val::Px(10.0)),
                border: UiRect {
                    left: Val::Px(3.0),
                    ..default()
                },
                border_radius: BorderRadius::all(Val::Px(10.0)),
                ..default()
            },
            BackgroundColor(OUTPUT_CARD_BG),
            BorderColor {
                left: PREVIEW_ACCENT,
                ..default()
            },
        ))
        .with_children(|card| {
            card.spawn((
                Text::new("Preview"),
                ui_font_semibold(13.0),
                TextColor(PREVIEW_ACCENT),
                Pickable::IGNORE,
            ));
            card.spawn((
                TypstPreviewPaper,
                Node {
                    width: Val::Percent(100.0),
                    flex_grow: 1.0,
                    flex_shrink: 1.0,
                    min_height: Val::Px(0.0),
                    align_items: AlignItems::Center,
                    justify_content: JustifyContent::Center,
                    overflow: Overflow::clip(),
                    border_radius: BorderRadius::all(Val::Px(8.0)),
                    ..default()
                },
                BackgroundColor(PREVIEW_PAPER),
            ))
            .with_children(|paper| {
                paper
                    .spawn((
                        TypstPreviewStatus,
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(0.0),
                            right: Val::Px(0.0),
                            top: Val::Px(0.0),
                            bottom: Val::Px(0.0),
                            padding: UiRect::all(Val::Px(10.0)),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            ..default()
                        },
                        Pickable::IGNORE,
                    ))
                    .with_children(|status| {
                        status.spawn((
                            Text::new("Rendered preview will appear here."),
                            result_font(14.0),
                            TextColor(Color::srgb(0.35, 0.33, 0.3)),
                            TypstPreviewStatusText,
                            Pickable::IGNORE,
                        ));
                    });
                paper.spawn((
                    TypstPreviewImage,
                    ImageNode {
                        image,
                        image_mode: NodeImageMode::Stretch,
                        ..default()
                    },
                    Node {
                        position_type: PositionType::Absolute,
                        width: Val::Px(1.0),
                        height: Val::Px(1.0),
                        display: Display::None,
                        ..default()
                    },
                    Pickable::IGNORE,
                ));
            });
        });
}

fn placeholder_image() -> Image {
    Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
}

#[allow(dead_code)]
fn spawn_results_scroll(parent: &mut ChildSpawnerCommands) {
    let mut scroll = parent.spawn((
        ResultsScroll,
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(8.0),
            width: Val::Percent(100.0),
            max_height: Val::Px(160.0),
            flex_shrink: 0.0,
            overflow: Overflow::scroll_y(),
            padding: UiRect::all(Val::Px(4.0)),
            ..default()
        },
        ScrollPosition::default(),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.15)),
    ));
    scroll.with_children(|col| {
        for task in TASKS {
            spawn_result_card(col, task);
        }
    });
    scroll.observe(scroll_overflow);
    stop_write_bubbling(&mut scroll);
}

fn spawn_output_card(parent: &mut ChildSpawnerCommands, kind: OutputKind) {
    parent
        .spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                width: Val::Percent(100.0),
                flex_grow: 1.0,
                flex_shrink: 1.0,
                min_height: Val::Px(96.0),
                padding: UiRect::all(Val::Px(10.0)),
                border: UiRect {
                    left: Val::Px(3.0),
                    ..default()
                },
                border_radius: BorderRadius::all(Val::Px(10.0)),
                ..default()
            },
            BackgroundColor(OUTPUT_CARD_BG),
            BorderColor {
                left: kind.accent(),
                ..default()
            },
        ))
        .with_children(|card| {
            card.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::Center,
                    width: Val::Percent(100.0),
                    column_gap: Val::Px(8.0),
                    ..default()
                },
            ))
            .with_children(|header| {
                header.spawn((
                    Text::new(kind.title()),
                    ui_font_semibold(13.0),
                    TextColor(kind.accent()),
                    Pickable::IGNORE,
                ));
                spawn_copy_button(header, kind);
            });

            let mut field = card.spawn((
                OutputField(kind),
                Node {
                    width: Val::Percent(100.0),
                    flex_grow: 1.0,
                    min_height: Val::Px(64.0),
                    padding: UiRect::all(Val::Px(8.0)),
                    overflow: Overflow::scroll_y(),
                    border_radius: BorderRadius::all(Val::Px(8.0)),
                    ..default()
                },
                BackgroundColor(FIELD_BG),
                EditableText {
                    visible_lines: None,
                    allow_newlines: true,
                    ..default()
                },
                TextLayout {
                    linebreak: LineBreak::WordOrCharacter,
                    ..default()
                },
                result_font(15.0),
                TextColor(Color::srgb(0.94, 0.95, 0.98)),
                TextCursorStyle {
                    color: Color::srgb(0.92, 0.94, 0.98),
                    selection_color: Color::srgba(0.35, 0.55, 0.95, 0.45),
                    unfocused_selection_color: Color::srgba(0.35, 0.55, 0.95, 0.22),
                    selected_text_color: Some(Color::WHITE),
                },
                TextScroll::default(),
            ));
            stop_write_bubbling(&mut field);
        });
}

fn spawn_copy_button(parent: &mut ChildSpawnerCommands, kind: OutputKind) {
    let mut button = parent.spawn((
        Button,
        CopyButton {
            kind,
            copied_until: 0.0,
        },
        Node {
            padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
            border: UiRect::all(Val::Px(1.0)),
            border_radius: BorderRadius::all(Val::Px(7.0)),
            ..default()
        },
        BackgroundColor(COPY_BG),
        BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.14)),
    ));
    button
        .with_children(|b| {
            b.spawn((
                Text::new("Copy"),
                ui_font_semibold(12.0),
                TextColor(Color::WHITE),
                CopyButtonLabel,
                Pickable::IGNORE,
            ));
        })
        .observe(
            |mut trigger: On<Pointer<Click>>,
             mut buttons: Query<&mut CopyButton>,
             recognized: Res<RecognizedText>,
             typst: Res<TypstText>,
             mut clipboard: ResMut<Clipboard>,
             time: Res<Time>| {
                trigger.propagate(false);
                let Ok(mut button) = buttons.get_mut(trigger.entity) else {
                    return;
                };
                let text = match button.kind {
                    OutputKind::Latex => recognized.0.as_str(),
                    OutputKind::Typst => typst.0.as_str(),
                };
                if let Err(err) = clipboard.set_text(text) {
                    eprintln!("clipboard: {err}");
                    return;
                }
                button.copied_until = time.elapsed_secs() + COPY_FLASH_SECS;
            },
        );
    stop_write_bubbling(&mut button);
}

#[allow(dead_code)]
#[allow(dead_code)]
fn spawn_result_card(parent: &mut ChildSpawnerCommands, task: &'static HunyuanTask) {
    parent
        .spawn((
            TaskResultCard(task.id),
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                width: Val::Percent(100.0),
                padding: UiRect::all(Val::Px(8.0)),
                border_radius: BorderRadius::all(Val::Px(8.0)),
                display: Display::None,
                ..default()
            },
            BackgroundColor(CARD_BG),
        ))
        .with_children(|card| {
            card.spawn((
                Text::new(format!("{}  —  {}", task.id, task.blurb)),
                ui_font(12.0),
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.7)),
            ));
            card.spawn((
                Text::new(""),
                result_font(14.0),
                TextColor(Color::WHITE),
                TaskResultBody(task.id),
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
                align_items: AlignItems::Stretch,
                width: Val::Percent(100.0),
                flex_shrink: 0.0,
                ..default()
            },
            GlobalZIndex(20),
        ))
        .with_children(|col| {
            col.spawn((Node {
                width: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                flex_shrink: 0.0,
                ..default()
            },))
                .with_children(|wrap| {
                    let mut header = wrap.spawn((
                        Button,
                        Node {
                            padding: UiRect::axes(Val::Px(10.0), Val::Px(6.0)),
                            width: Val::Percent(100.0),
                            border_radius: BorderRadius::all(Val::Px(8.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.14)),
                    ));
                    header
                        .with_children(|b| {
                            b.spawn((
                                Text::new(""),
                                ui_font(15.0),
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

                    // Out of flow: closed picker is one header row; open it
                    // overlays the transcription cards instead of shoving them down.
                    let mut list = wrap.spawn((
                        TaskMenuList,
                        Node {
                            position_type: PositionType::Absolute,
                            top: Val::Percent(100.0),
                            left: Val::Px(0.0),
                            right: Val::Px(0.0),
                            flex_direction: FlexDirection::Column,
                            row_gap: Val::Px(2.0),
                            padding: UiRect::all(Val::Px(4.0)),
                            max_height: Val::Vh(40.0),
                            overflow: Overflow::scroll_y(),
                            border: UiRect::all(Val::Px(1.0)),
                            border_radius: BorderRadius::all(Val::Px(8.0)),
                            display: Display::None,
                            ..default()
                        },
                        ScrollPosition::default(),
                        BackgroundColor(MENU_BG),
                        BorderColor::all(PANEL_BORDER),
                        GlobalZIndex(110),
                    ));
                    list.with_children(|list| {
                        for task in TASKS {
                            spawn_task_option(list, task);
                        }
                    });
                    list.observe(scroll_overflow);
                    stop_write_bubbling(&mut list);
                });

            col.spawn((
                Text::new(""),
                ui_font(12.0),
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.55)),
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
                ui_font(13.0),
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

fn scroll_overflow(
    on_scroll: On<Pointer<Scroll>>,
    mut query: Query<(&mut ScrollPosition, &ComputedNode)>,
) {
    let Ok((mut scroll_position, node)) = query.get_mut(on_scroll.observer()) else {
        return;
    };
    let dy = match on_scroll.unit {
        MouseScrollUnit::Line => on_scroll.y * 24.0,
        MouseScrollUnit::Pixel => on_scroll.y,
    };
    let range = (node.content_size.y - node.size.y).max(0.0) * node.inverse_scale_factor;
    scroll_position.y = (scroll_position.y - dy).clamp(0.0, range);
}

fn unfocus_when_writing(
    canvas: Query<&CellInk, With<TestCanvas>>,
    mut focus: ResMut<InputFocus>,
    fields: Query<Entity, With<OutputField>>,
) {
    let Ok(cell) = canvas.single() else {
        return;
    };
    if !cell.is_writing() {
        return;
    }
    if focus.get().is_some_and(|entity| fields.get(entity).is_ok()) {
        focus.clear();
    }
}

fn output_field_focused(focus: &InputFocus, fields: &Query<Entity, With<OutputField>>) -> bool {
    focus.get().is_some_and(|entity| fields.get(entity).is_ok())
}

fn show_ui(mut roots: Query<&mut Visibility, With<TestSplitRoot>>) {
    for mut vis in &mut roots {
        *vis = Visibility::Inherited;
    }
}

fn hide_ui(mut roots: Query<&mut Visibility, With<TestSplitRoot>>, mut state: ResMut<VlTaskState>) {
    state.menu_open = false;
    for mut vis in &mut roots {
        *vis = Visibility::Hidden;
    }
}

fn clear_canvas(
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut pending: ResMut<PendingOcr>,
    mut state: ResMut<VlTaskState>,
    mut recognized: ResMut<RecognizedText>,
    mut scroll: Query<&mut ScrollPosition, With<ResultsScroll>>,
    ocr: Option<NonSendMut<OcrClient>>,
    writing: Res<UiPointerDown>,
) {
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
    for mut pos in &mut scroll {
        pos.0 = Vec2::ZERO;
    }
}

fn clear_on_key(
    keys: Res<ButtonInput<KeyCode>>,
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut pending: ResMut<PendingOcr>,
    mut state: ResMut<VlTaskState>,
    mut recognized: ResMut<RecognizedText>,
    mut scroll: Query<&mut ScrollPosition, With<ResultsScroll>>,
    ocr: Option<NonSendMut<OcrClient>>,
    writing: Res<UiPointerDown>,
    mut focus: ResMut<InputFocus>,
    fields: Query<Entity, With<OutputField>>,
) {
    if output_field_focused(&focus, &fields) {
        if keys.just_pressed(KeyCode::Escape) {
            focus.clear();
        }
        return;
    }
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
        for mut pos in &mut scroll {
            pos.0 = Vec2::ZERO;
        }
    }
}

fn update_recognized_text(
    mut canvas: Query<&mut CellInk, With<TestCanvas>>,
    mut recognized: ResMut<RecognizedText>,
    mut pending: ResMut<PendingOcr>,
    mut state: ResMut<VlTaskState>,
    ocr: Option<NonSendMut<OcrClient>>,
    time: Res<Time>,
) {
    let Ok(mut cell) = canvas.single_mut() else {
        return;
    };
    let Some(mut ocr) = ocr else { return };

    while let Some((id, current, result)) = ocr.poll() {
        if ocr.is_vlm() {
            if let Some(task) = state.submitted.remove(&id) {
                apply_vlm_result(result, &task, &mut state, &mut recognized);
            }
        } else if current {
            apply_hat_result(result, &mut recognized);
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
        if cell.ink.is_empty() {
            pending.ink = None;
            state.forget_ink();
            recognized.0.clear();
            ocr.cancel();
            return;
        }
        if ocr.is_vlm() {
            pending.ink = Some(cell.ink.clone());
            pending.due = time.elapsed_secs() + VLM_DEBOUNCE;
        } else {
            pending.ink = None;
            ocr.submit(cell.ink.clone(), None);
            recognized.0 = "recognizing…".to_string();
        }
    }

    if ocr.is_vlm() && state.rerun {
        state.rerun = false;
        if pending.ink.is_none() {
            if state.results.contains_key(&state.selected) {
                show_cached(&state, &mut recognized);
            } else if let Some(ink) = state.last_ink.clone() {
                submit_vlm(&mut ocr, ink, &mut state, &mut recognized);
            }
        }
    }

    if pending.ink.is_some() && !cell.is_writing() && time.elapsed_secs() >= pending.due {
        let ink = pending.ink.take().expect("pending ink was Some");
        if ocr.is_vlm() {
            state.last_ink = Some(ink.clone());
            state.results.clear();
            submit_vlm(&mut ocr, ink, &mut state, &mut recognized);
        } else {
            ocr.submit(ink, None);
            recognized.0 = "recognizing…".to_string();
        }
    }
}

fn submit_vlm(
    ocr: &mut OcrClient,
    ink: Ink,
    state: &mut VlTaskState,
    recognized: &mut RecognizedText,
) {
    let prompt = hunyuan_tasks::prompt_for(&state.selected)
        .unwrap_or_else(|| hunyuan_tasks::prompt_for(hunyuan_tasks::DEFAULT_TASK_ID).unwrap())
        .to_string();
    let task = state.selected.clone();
    let id = ocr.submit(ink, Some(prompt));
    state.submitted.insert(id, task.clone());
    state.status = format!("recognizing {task}…");
    recognized.0 = state.status.clone();
}

fn apply_vlm_result(
    result: Result<String, String>,
    task: &str,
    state: &mut VlTaskState,
    recognized: &mut RecognizedText,
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
    if state.submitted.is_empty() {
        state.status.clear();
    }
    if task == state.selected {
        show_cached(state, recognized);
    }
}

fn show_cached(state: &VlTaskState, recognized: &mut RecognizedText) {
    recognized.0 = state
        .results
        .get(&state.selected)
        .cloned()
        .unwrap_or_else(|| state.status.clone());
}

fn apply_hat_result(result: Result<String, String>, recognized: &mut RecognizedText) {
    match result {
        Ok(text) => recognized.0 = text,
        Err(err) => eprintln!("recognition failed: {err}"),
    }
}

fn sync_task_dropdown(
    state: Res<VlTaskState>,
    mut header: Query<&mut Text, With<TaskMenuHeaderLabel>>,
    mut preview: Query<&mut Text, (With<TaskPromptPreview>, Without<TaskMenuHeaderLabel>)>,
    mut list: Query<&mut Node, With<TaskMenuList>>,
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
    let prompt = task.map(|task| task.prompt_en).unwrap_or("");
    for mut text in &mut preview {
        text.0 = prompt.to_string();
    }
    for mut node in &mut list {
        node.display = if state.menu_open {
            Display::Flex
        } else {
            Display::None
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

fn sync_result_ui(
    recognized: Res<RecognizedText>,
    mut typst: ResMut<TypstText>,
    state: Res<VlTaskState>,
    mut fields: Query<(&OutputField, &mut EditableText)>,
    mut cards: Query<(&TaskResultCard, &mut Node, &mut BackgroundColor)>,
    mut bodies: Query<(&TaskResultBody, &mut Text)>,
) {
    let current = if recognized.0.is_empty() {
        String::new()
    } else {
        recognized.0.clone()
    };
    if recognized.is_changed() {
        typst.0 = latex_to_typst(&current);
    }
    for (field, mut editable) in &mut fields {
        let next = match field.0 {
            OutputKind::Latex => current.as_str(),
            OutputKind::Typst => typst.0.as_str(),
        };
        set_editable_text(&mut editable, next);
    }

    for (card, mut node, mut bg) in &mut cards {
        let has = state.results.contains_key(card.0);
        node.display = if has {
            Display::Flex
        } else {
            Display::None
        };
        *bg = BackgroundColor(if card.0 == state.selected {
            CARD_SELECTED_BG
        } else {
            CARD_BG
        });
    }
    for (body, mut text) in &mut bodies {
        text.0 = state.results.get(body.0).cloned().unwrap_or_default();
    }
}

fn set_editable_text(editable: &mut EditableText, next: &str) {
    if editable.value().to_string() == next {
        return;
    }
    editable.editor.set_text(next);
    editable.pending_edits.clear();
    editable.pending_paste = None;
}

fn sync_typst_preview(
    typst: Res<TypstText>,
    mut client: NonSendMut<TypstPreviewClient>,
    texture: Res<PreviewTexture>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut statuses: Query<&mut Node, (With<TypstPreviewStatus>, Without<TypstPreviewImage>)>,
    mut status_texts: Query<&mut Text, With<TypstPreviewStatusText>>,
    mut previews: Query<&mut Node, (With<TypstPreviewImage>, Without<TypstPreviewStatus>)>,
) {
    let scale = windows
        .single()
        .map(|window| window.scale_factor())
        .unwrap_or(1.0);
    // Bevy UI images are shown in logical pixels, then multiplied by the
    // window scale; render extra physical pixels so the preview stays sharp.
    let pixel_per_pt = (3.0 * scale).clamp(2.0, 8.0);
    client.submit(typst.0.clone(), pixel_per_pt);
    let Some(result) = client.poll() else {
        return;
    };
    match result {
        PreviewResult::Empty => {
            for mut text in &mut status_texts {
                text.0 = "Rendered preview will appear here.".to_string();
            }
            for mut node in &mut statuses {
                node.display = Display::Flex;
            }
            for mut node in &mut previews {
                node.display = Display::None;
            }
        }
        PreviewResult::Error(err) => {
            for mut text in &mut status_texts {
                text.0 = err.clone();
            }
            for mut node in &mut statuses {
                node.display = Display::Flex;
            }
            for mut node in &mut previews {
                node.display = Display::None;
            }
        }
        PreviewResult::Frame(frame) => {
            if let Some(mut image) = images.get_mut(&texture.0) {
                *image = Image::new(
                    Extent3d {
                        width: frame.width,
                        height: frame.height,
                        depth_or_array_layers: 1,
                    },
                    TextureDimension::D2,
                    frame.rgba,
                    TextureFormat::Rgba8UnormSrgb,
                    RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
                );
            }
            for mut node in &mut statuses {
                node.display = Display::None;
            }
            for mut node in &mut previews {
                node.display = Display::Flex;
            }
        }
    }
}

fn fit_typst_preview_image(
    papers: Query<&ComputedNode, With<TypstPreviewPaper>>,
    mut images: Query<&mut Node, With<TypstPreviewImage>>,
    texture: Res<PreviewTexture>,
    assets: Res<Assets<Image>>,
) {
    let Ok(paper) = papers.single() else {
        return;
    };
    let Ok(mut node) = images.single_mut() else {
        return;
    };
    if node.display == Display::None {
        return;
    }
    let Some(image) = assets.get(&texture.0) else {
        return;
    };
    let tex = image.size();
    if tex.x == 0 || tex.y == 0 {
        return;
    }

    let inset = 8.0;
    let max_w = (paper.size.x * paper.inverse_scale_factor - 2.0 * inset).max(1.0);
    let max_h = (paper.size.y * paper.inverse_scale_factor - 2.0 * inset).max(1.0);
    let aspect = tex.x as f32 / tex.y as f32;
    let (width, height) = if max_w / aspect <= max_h {
        (max_w, max_w / aspect)
    } else {
        (max_h * aspect, max_h)
    };
    let left = ((paper.size.x * paper.inverse_scale_factor - width) * 0.5).max(0.0);
    let top = ((paper.size.y * paper.inverse_scale_factor - height) * 0.5).max(0.0);

    node.position_type = PositionType::Absolute;
    node.width = Val::Px(width);
    node.height = Val::Px(height);
    node.left = Val::Px(left);
    node.top = Val::Px(top);
}

fn sync_copy_buttons(
    time: Res<Time>,
    mut buttons: Query<(&CopyButton, &mut BackgroundColor, &Children)>,
    mut labels: Query<&mut Text, With<CopyButtonLabel>>,
) {
    let now = time.elapsed_secs();
    for (button, mut bg, children) in &mut buttons {
        let copied = button.copied_until > now;
        *bg = BackgroundColor(if copied { COPY_FLASH_BG } else { COPY_BG });
        for child in children {
            if let Ok(mut text) = labels.get_mut(*child) {
                text.0 = if copied {
                    "Copied".to_string()
                } else {
                    "Copy".to_string()
                };
            }
        }
    }
}
