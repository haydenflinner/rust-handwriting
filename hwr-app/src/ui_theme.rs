//! Shared UI type styles. Bevy's default font is a Latin subset of Fira Mono,
//! which tofu's Hunyuan's Chinese prompts and a lot of the model output.

use bevy::prelude::*;

pub fn ui_font(size: f32) -> TextFont {
    TextFont {
        font: FontSource::SystemUi,
        font_size: FontSize::Px(size),
        ..default()
    }
}

pub fn result_font(size: f32) -> TextFont {
    // System UI, not UiMonospace: SF Mono / Menlo lack CJK, and Hunyuan
    // often mixes English code with Chinese task wrappers in one string.
    TextFont {
        font: FontSource::SystemUi,
        font_size: FontSize::Px(size),
        ..default()
    }
}

pub fn ui_font_semibold(size: f32) -> TextFont {
    TextFont {
        font: FontSource::SystemUi,
        font_size: FontSize::Px(size),
        weight: FontWeight::SEMIBOLD,
        ..default()
    }
}
