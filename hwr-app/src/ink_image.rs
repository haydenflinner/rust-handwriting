//! Rasterize online ink to an RGB document crop for vision-language OCR.
//!
//! VLMs expect dark ink on a light page, not the app's light-on-dark canvas,
//! and they want a much larger crop than HAT's 224² offline branch.

use hwr_ink::ink::Ink;
use image::{Rgb, RgbImage};

/// Long side of the letterboxed crop. Hunyuan's processor still smart-resizes
/// toward `min_pixels` / `max_pixels`; 1280 made a few handwritten words into
/// a huge vision-token prefill. 768 stays sharp for a line of ink.
const LONG_SIDE: u32 = 768;
const MARGIN: f32 = 48.0;
const MIN_SHORT_SIDE: f32 = 96.0;

/// Dark-on-light document colors (not the on-screen gizmo colors).
const PAPER: Rgb<u8> = Rgb([255, 255, 255]);
const INK: Rgb<u8> = Rgb([20, 20, 24]);

pub fn rasterize_ink_rgb(ink: &Ink) -> Option<RgbImage> {
    if ink.is_empty() {
        return None;
    }

    let min_x = ink.x_range.min;
    let min_y = ink.y_range.min;
    let dx = (ink.x_range.max - min_x).max(1.0);
    let dy = (ink.y_range.max - min_y).max(1.0);

    let usable = LONG_SIDE as f32 - 2.0 * MARGIN;
    let scale = usable / dx.max(dy);

    let content_w = dx * scale;
    let content_h = dy * scale;
    let width = (content_w + 2.0 * MARGIN)
        .max(MIN_SHORT_SIDE)
        .round()
        .clamp(32.0, LONG_SIDE as f32) as u32;
    let height = (content_h + 2.0 * MARGIN)
        .max(MIN_SHORT_SIDE)
        .round()
        .clamp(32.0, LONG_SIDE as f32) as u32;

    let origin_x = (width as f32 - content_w) * 0.5;
    let origin_y = (height as f32 - content_h) * 0.5;
    let mut img = RgbImage::from_pixel(width, height, PAPER);

    let map = |x: f32, y: f32| -> (i32, i32) {
        let px = origin_x + (x - min_x) * scale;
        let py = origin_y + (y - min_y) * scale;
        (px.round() as i32, py.round() as i32)
    };

    let radius = ((scale * 1.25).round() as i32).clamp(2, 6);
    for stroke in ink.strokes() {
        if stroke.is_empty() {
            continue;
        }
        if stroke.len() == 1 {
            let (x, y) = map(stroke[0].x, stroke[0].y);
            stamp(&mut img, x, y, radius);
            continue;
        }
        for pair in stroke.windows(2) {
            let (x0, y0) = map(pair[0].x, pair[0].y);
            let (x1, y1) = map(pair[1].x, pair[1].y);
            draw_line(&mut img, x0, y0, x1, y1, radius);
        }
    }

    Some(img)
}

fn stamp(img: &mut RgbImage, x: i32, y: i32, radius: i32) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let r2 = radius * radius;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > r2 {
                continue;
            }
            let xx = x + dx;
            let yy = y + dy;
            if xx >= 0 && yy >= 0 && xx < w && yy < h {
                img.put_pixel(xx as u32, yy as u32, INK);
            }
        }
    }
}

fn draw_line(img: &mut RgbImage, mut x0: i32, mut y0: i32, x1: i32, y1: i32, radius: i32) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        stamp(img, x0, y0, radius);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}
