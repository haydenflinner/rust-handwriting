//! Scratch-out: a zigzag scribble over existing ink deletes the letters
//! it covers, instead of adding another stroke.
//!
//! Detection is conservative so ordinary glyphs (`m`, `w`, `E`) are not
//! mistaken for erasers: the new stroke must reverse direction several
//! times *and* spend a large fraction of its length near earlier ink.
//! Multi-stroke letters (`t`, `i`, `=`) are clustered by nearby bounding
//! boxes so scribbling the stem also takes the crossbar.

use cgmath::{InnerSpace, MetricSpace, Point2, Point3, Vector2};

use crate::ink::Ink;
use crate::math::xy;

impl Ink {
    /// If the most recently closed stroke is a scribble over earlier ink,
    /// drop that stroke and remove the letter-clusters it covered.
    ///
    /// Call after `pen_up()`. Returns whether a scratch-out was applied.
    pub fn apply_scratch_out(&mut self) -> bool {
        let mut strokes: Vec<Vec<Point3<f32>>> = self.strokes().map(|s| s.to_vec()).collect();
        if strokes.len() < 2 {
            return false;
        }
        let scribble = strokes.pop().expect("len >= 2");
        if !stroke_is_scratch_out_against(&scribble, &strokes) {
            return false;
        }

        let existing_pts: Vec<Point3<f32>> = strokes.iter().flatten().copied().collect();
        let radius = scratch_radius_for_points(&existing_pts);
        let hit = hit_strokes(&strokes, &scribble, radius);
        let drop = expand_hits_to_clusters(&strokes, &hit);

        if hit.iter().any(|&h| h) {
            let mut result = Ink::new();
            for (i, stroke) in strokes.into_iter().enumerate() {
                if !drop[i] {
                    push_stroke(&mut result, &stroke);
                }
            }
            *self = result;
            return true;
        }

        // Classified as a scribble but no whole stroke passed the hit test —
        // punch a hole so the gesture still erases the ink it covered.
        let mut earlier = Ink::new();
        for stroke in &strokes {
            push_stroke(&mut earlier, stroke);
        }
        earlier.erase(&ink_from_points(&scribble), radius);
        *self = earlier;
        true
    }

    /// True when the in-progress (not yet `pen_up`) stroke already looks
    /// like a scratch-out over the closed strokes. Used to preview the
    /// scribble in an erase color while the pen is down.
    pub fn preview_scratch_out(&self) -> bool {
        let last_closed = self.stroke_ends.last().copied().unwrap_or(0);
        if last_closed == 0 || last_closed >= self.points.len() {
            return false;
        }
        let existing: Vec<Vec<Point3<f32>>> = self.strokes().map(|s| s.to_vec()).collect();
        stroke_is_scratch_out_against(&self.points[last_closed..], &existing)
    }
}

fn ink_from_points(points: &[Point3<f32>]) -> Ink {
    let mut ink = Ink::new();
    push_stroke(&mut ink, points);
    ink
}

fn push_stroke(ink: &mut Ink, points: &[Point3<f32>]) {
    if points.is_empty() {
        return;
    }
    for p in points {
        ink.push(p.x, p.y, p.z);
    }
    ink.pen_up();
}

fn scratch_radius_for_points(points: &[Point3<f32>]) -> f32 {
    if points.is_empty() {
        return 12.0;
    }
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    for p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let span = (max_y - min_y).max(max_x - min_x);
    (span * 0.18).clamp(8.0, 40.0)
}

fn stroke_is_scratch_out_against(scribble: &[Point3<f32>], existing: &[Vec<Point3<f32>>]) -> bool {
    if existing.is_empty() || scribble.len() < 4 {
        return false;
    }
    let existing_pts: Vec<Point3<f32>> = existing.iter().flatten().copied().collect();
    if existing_pts.is_empty() {
        return false;
    }
    let radius = scratch_radius_for_points(&existing_pts);
    let min_seg = (radius * 0.7).clamp(6.0, 24.0);
    if direction_reversals(scribble, min_seg) < 3 {
        return false;
    }
    let existing_strokes: Vec<&[Point3<f32>]> = existing.iter().map(|s| s.as_slice()).collect();
    if fraction_near(scribble, &existing_strokes, radius) < 0.35 {
        return false;
    }
    let path_len = polyline_len(scribble);
    let (w, h) = bbox_size(scribble);
    let span = w.max(h).max(1.0);
    path_len / span >= 1.6
}

fn direction_reversals(stroke: &[Point3<f32>], min_seg: f32) -> usize {
    if stroke.len() < 3 {
        return 0;
    }
    let mut reversals = 0;
    let mut prev_dir: Option<Vector2<f32>> = None;
    let mut acc = Vector2::new(0.0, 0.0);
    let mut last = xy(stroke[0]);
    for p in &stroke[1..] {
        let q = xy(*p);
        acc += q - last;
        last = q;
        if acc.magnitude() < min_seg {
            continue;
        }
        if let Some(prev) = prev_dir {
            if prev.dot(acc) < 0.0 {
                reversals += 1;
            }
        }
        prev_dir = Some(acc);
        acc = Vector2::new(0.0, 0.0);
    }
    reversals
}

fn polyline_len(stroke: &[Point3<f32>]) -> f32 {
    stroke
        .windows(2)
        .map(|pair| xy(pair[0]).distance(xy(pair[1])))
        .sum()
}

fn bbox_size(stroke: &[Point3<f32>]) -> (f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for p in stroke {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    ((max_x - min_x).max(0.0), (max_y - min_y).max(0.0))
}

fn bbox(stroke: &[Point3<f32>]) -> (f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for p in stroke {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    (min_x, min_y, max_x, max_y)
}

fn fraction_near(scribble: &[Point3<f32>], existing: &[&[Point3<f32>]], radius: f32) -> f32 {
    if scribble.is_empty() {
        return 0.0;
    }
    let radius2 = radius * radius;
    let near = scribble
        .iter()
        .filter(|p| existing.iter().any(|stroke| point_near_stroke(xy(**p), stroke, radius2)))
        .count();
    near as f32 / scribble.len() as f32
}

fn point_near_stroke(q: Point2<f32>, stroke: &[Point3<f32>], radius2: f32) -> bool {
    if stroke.is_empty() {
        return false;
    }
    if stroke.len() == 1 {
        return xy(stroke[0]).distance2(q) <= radius2;
    }
    stroke.windows(2).any(|pair| {
        point_segment_distance2(xy(pair[0]), xy(pair[1]), q) <= radius2
    })
}

fn point_segment_distance2(p0: Point2<f32>, p1: Point2<f32>, q: Point2<f32>) -> f32 {
    if p0 == p1 {
        return p0.distance2(q);
    }
    let u: Vector2<f32> = p1 - p0;
    let v: Vector2<f32> = q - p0;
    let t = u.dot(v) / u.magnitude2();
    if t <= 0.0 {
        p0.distance2(q)
    } else if t >= 1.0 {
        p1.distance2(q)
    } else {
        (p0 + u * t).distance2(q)
    }
}

fn hit_strokes(strokes: &[Vec<Point3<f32>>], scribble: &[Point3<f32>], radius: f32) -> Vec<bool> {
    let radius2 = radius * radius;
    strokes
        .iter()
        .map(|stroke| {
            if stroke.is_empty() {
                return false;
            }
            let near = stroke
                .iter()
                .filter(|p| point_near_stroke(xy(**p), scribble, radius2))
                .count();
            let frac = near as f32 / stroke.len() as f32;
            near >= 3 || frac >= 0.2
        })
        .collect()
}

fn expand_hits_to_clusters(strokes: &[Vec<Point3<f32>>], hit: &[bool]) -> Vec<bool> {
    let n = strokes.len();
    if n == 0 {
        return Vec::new();
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let a = find(parent, a);
        let b = find(parent, b);
        if a != b {
            parent[b] = a;
        }
    }

    let boxes: Vec<_> = strokes.iter().map(|s| bbox(s)).collect();
    for i in 0..n {
        let (min_x, min_y, max_x, max_y) = boxes[i];
        let hi = (max_y - min_y).max(8.0);
        for j in (i + 1)..n {
            let (omin_x, omin_y, omax_x, omax_y) = boxes[j];
            let hj = (omax_y - omin_y).max(8.0);
            let pad = 0.25 * hi.max(hj);
            let overlap = min_x - pad <= omax_x
                && omin_x - pad <= max_x
                && min_y - pad <= omax_y
                && omin_y - pad <= max_y;
            if overlap {
                union(&mut parent, i, j);
            }
        }
    }

    let mut drop = vec![false; n];
    for i in 0..n {
        if !hit[i] {
            continue;
        }
        let root = find(&mut parent, i);
        for j in 0..n {
            if find(&mut parent, j) == root {
                drop[j] = true;
            }
        }
    }
    drop
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vertical_letter(x: f32) -> Ink {
        let mut ink = Ink::new();
        for i in 0..=20 {
            ink.push(x, i as f32 * 5.0, i as f32 * 0.05);
        }
        ink.pen_up();
        ink
    }

    fn push_zigzag(ink: &mut Ink, cx: f32) {
        let mut t = ink.t_range.max;
        if !t.is_finite() {
            t = 0.0;
        }
        t += 0.5;
        // True back-and-forth (not a 90° staircase): each new point flips
        // x so consecutive segments reverse direction.
        for i in 0..6 {
            let y = 15.0 + i as f32 * 14.0;
            let x = if i % 2 == 0 { cx - 12.0 } else { cx + 12.0 };
            ink.push(x, y, t);
            t += 0.04;
        }
        ink.pen_up();
    }

    #[test]
    fn scribble_erases_the_letter_underneath() {
        let mut ink = vertical_letter(10.0);
        push_zigzag(&mut ink, 10.0);
        assert!(ink.apply_scratch_out());
        assert!(ink.is_empty(), "letter and scribble should both be gone");
    }

    #[test]
    fn scribble_over_one_letter_leaves_the_neighbor() {
        let mut ink = vertical_letter(10.0);
        ink.append(vertical_letter(120.0), 0.5);
        push_zigzag(&mut ink, 10.0);
        assert!(ink.apply_scratch_out());
        assert_eq!(ink.strokes().count(), 1);
        let remaining: Vec<_> = ink.strokes().next().unwrap().iter().map(|p| p.x).collect();
        assert!(
            remaining.iter().all(|x| *x > 100.0),
            "the far letter should remain"
        );
    }

    #[test]
    fn scribble_on_empty_canvas_is_kept() {
        let mut ink = Ink::new();
        push_zigzag(&mut ink, 10.0);
        assert!(!ink.apply_scratch_out());
        assert!(!ink.is_empty());
    }

    #[test]
    fn a_second_plain_stroke_is_not_a_scribble() {
        let mut ink = vertical_letter(10.0);
        ink.append(vertical_letter(80.0), 0.5);
        assert!(!ink.apply_scratch_out());
        assert_eq!(ink.strokes().count(), 2);
    }

    #[test]
    fn scribble_on_t_removes_stem_and_crossbar() {
        let mut ink = Ink::new();
        for i in 0..=20 {
            ink.push(40.0, i as f32 * 5.0, i as f32 * 0.05);
        }
        ink.pen_up();
        ink.push(20.0, 30.0, 2.0);
        ink.push(60.0, 30.0, 2.1);
        ink.pen_up();
        push_zigzag(&mut ink, 40.0);
        assert!(ink.apply_scratch_out());
        assert!(
            ink.is_empty(),
            "stem and crossbar are one letter and should both go"
        );
    }

    #[test]
    fn preview_turns_true_once_the_zigzag_is_established() {
        let mut ink = vertical_letter(10.0);
        assert!(!ink.preview_scratch_out());
        // Open stroke: same zigzag, no pen_up yet.
        let mut t = 2.0;
        for i in 0..6 {
            let y = 15.0 + i as f32 * 14.0;
            let x = if i % 2 == 0 { 0.0 } else { 20.0 };
            ink.push(x, y, t);
            t += 0.04;
        }
        assert!(ink.preview_scratch_out());
    }
}
