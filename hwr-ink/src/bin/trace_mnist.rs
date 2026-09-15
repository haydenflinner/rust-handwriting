//! Trace MNIST digit bitmaps into armrest-format ink vectors.
//!
//! Offline 28×28 grayscale is upsampled, closed, Zhang–Suen thinned, then
//! walked as an 8-connected skeleton into strokes. Timing is synthetic
//! (constant pen speed plus a small lift between strokes) — these are
//! geometric traces, not captured pen dynamics. `spline::prepare` still
//! height-normalizes them the same way as real ink.
//!
//! Usage:
//!   trace_mnist --images PATH --labels PATH --out PATH
//!               [--limit N] [--previews DIR] [--preview-per-class N]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use hwr_ink::ink::Ink;
use hwr_ink::math::douglas_peucker;

const SRC: usize = 28;
const SCALE: usize = 3;
const DST: usize = SRC * SCALE;
const BIN_THRESHOLD: u8 = 90;
const SPUR_LEN: usize = 4;
const DP_EPS: f32 = 0.85;
const RESAMPLE: f32 = 0.9;
const SPEED: f32 = 70.0;
const PEN_LIFT: f32 = 0.08;

fn main() {
    let mut images_path: Option<PathBuf> = None;
    let mut labels_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut previews: Option<PathBuf> = None;
    let mut preview_per_class: usize = 2;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--images" => images_path = Some(PathBuf::from(args.next().expect("--images needs a path"))),
            "--labels" => labels_path = Some(PathBuf::from(args.next().expect("--labels needs a path"))),
            "--out" => out_path = Some(PathBuf::from(args.next().expect("--out needs a path"))),
            "--limit" => {
                limit = Some(
                    args.next()
                        .expect("--limit needs a number")
                        .parse()
                        .expect("limit must be a number"),
                )
            }
            "--previews" => previews = Some(PathBuf::from(args.next().expect("--previews needs a path"))),
            "--preview-per-class" => {
                preview_per_class = args
                    .next()
                    .expect("--preview-per-class needs a number")
                    .parse()
                    .expect("preview-per-class must be a number")
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let images_path = images_path.expect("--images is required");
    let labels_path = labels_path.expect("--labels is required");
    let out_path = out_path.expect("--out is required");

    let (n, images) = read_idx_images(&images_path).expect("read images");
    let labels = read_idx_labels(&labels_path).expect("read labels");
    if labels.len() != n {
        panic!(
            "image/label count mismatch: {} images, {} labels",
            n,
            labels.len()
        );
    }

    let n = limit.map(|l| l.min(n)).unwrap_or(n);
    if let Some(dir) = &previews {
        std::fs::create_dir_all(dir).expect("create previews dir");
    }

    let mut out = BufWriter::new(File::create(&out_path).expect("create output"));
    let mut previewed = [0usize; 10];
    let mut per_class = [0usize; 10];
    let mut skipped = 0usize;
    let mut stroke_sum = 0usize;
    let mut point_sum = 0usize;

    for i in 0..n {
        let src = &images[i * SRC * SRC..(i + 1) * SRC * SRC];
        let label = labels[i];
        let Some(ink) = trace_digit(src) else {
            skipped += 1;
            continue;
        };
        let class = label as usize;
        if class < 10 {
            per_class[class] += 1;
        }
        stroke_sum += ink.strokes().count();
        point_sum += ink.len();

        if let Some(dir) = &previews {
            if class < 10 && previewed[class] < preview_per_class {
                let name = dir.join(format!("class{label}_{:05}.bmp", previewed[class]));
                write_preview_bmp(&name, src, &ink).expect("write preview");
                previewed[class] += 1;
            }
        }

        writeln!(out, "{label}\t{ink}").expect("write sample");

        if (i + 1) % 5000 == 0 {
            println!("traced {}/{n}", i + 1);
        }
    }
    out.flush().expect("flush");

    let kept = n - skipped;
    println!(
        "Wrote {kept} traces to {} (skipped {skipped} empty). mean strokes={:.2} mean points={:.1}",
        out_path.display(),
        if kept > 0 {
            stroke_sum as f32 / kept as f32
        } else {
            0.0
        },
        if kept > 0 {
            point_sum as f32 / kept as f32
        } else {
            0.0
        }
    );
    print!("per class:");
    for (d, c) in per_class.iter().enumerate() {
        print!(" {d}:{c}");
    }
    println!();
}

fn read_u32_be(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_idx_images(path: &Path) -> io::Result<(usize, Vec<u8>)> {
    let mut f = File::open(path)?;
    let magic = read_u32_be(&mut f)?;
    if magic != 2051 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad image magic {magic}"),
        ));
    }
    let n = read_u32_be(&mut f)? as usize;
    let rows = read_u32_be(&mut f)? as usize;
    let cols = read_u32_be(&mut f)? as usize;
    if rows != SRC || cols != SRC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected {SRC}x{SRC}, got {rows}x{cols}"),
        ));
    }
    let mut data = vec![0u8; n * SRC * SRC];
    f.read_exact(&mut data)?;
    Ok((n, data))
}

fn read_idx_labels(path: &Path) -> io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let magic = read_u32_be(&mut f)?;
    if magic != 2049 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad label magic {magic}"),
        ));
    }
    let n = read_u32_be(&mut f)? as usize;
    let mut data = vec![0u8; n];
    f.read_exact(&mut data)?;
    Ok(data)
}

fn trace_digit(src: &[u8]) -> Option<Ink> {
    let gray = upsample_bilinear(src);
    let mut bin = vec![0u8; DST * DST];
    for (i, &g) in gray.iter().enumerate() {
        if g >= BIN_THRESHOLD {
            bin[i] = 1;
        }
    }
    morphological_close(&mut bin, DST, DST);
    zhang_suen(&mut bin, DST, DST);
    prune_spurs(&mut bin, DST, DST, SPUR_LEN);

    let mut paths = extract_paths(&bin, DST, DST);
    if paths.is_empty() {
        return None;
    }
    order_paths(&mut paths);

    let mut ink = Ink::new();
    let mut t = 0.0f32;
    for path in &paths {
        if path.is_empty() {
            continue;
        }
        let mut prev = path[0];
        ink.push(prev.0 as f32, prev.1 as f32, t);
        for &p in &path[1..] {
            let dx = p.0 as f32 - prev.0 as f32;
            let dy = p.1 as f32 - prev.1 as f32;
            t += (dx * dx + dy * dy).sqrt() / SPEED;
            ink.push(p.0 as f32, p.1 as f32, t);
            prev = p;
        }
        ink.pen_up();
        t += PEN_LIFT;
    }
    if ink.is_empty() {
        return None;
    }
    let ink = douglas_peucker(&ink, DP_EPS);
    Some(ink.resample(RESAMPLE))
}

fn upsample_bilinear(src: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; DST * DST];
    for y in 0..DST {
        let fy = (y as f32 + 0.5) / SCALE as f32 - 0.5;
        let y0 = fy.floor().clamp(0.0, (SRC - 1) as f32) as usize;
        let y1 = (y0 + 1).min(SRC - 1);
        let ty = fy - y0 as f32;
        for x in 0..DST {
            let fx = (x as f32 + 0.5) / SCALE as f32 - 0.5;
            let x0 = fx.floor().clamp(0.0, (SRC - 1) as f32) as usize;
            let x1 = (x0 + 1).min(SRC - 1);
            let tx = fx - x0 as f32;
            let v00 = src[y0 * SRC + x0] as f32;
            let v10 = src[y0 * SRC + x1] as f32;
            let v01 = src[y1 * SRC + x0] as f32;
            let v11 = src[y1 * SRC + x1] as f32;
            let v0 = v00 + (v10 - v00) * tx;
            let v1 = v01 + (v11 - v01) * tx;
            out[y * DST + x] = (v0 + (v1 - v0) * ty).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

fn idx(x: i32, y: i32, w: usize, h: usize) -> Option<usize> {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        None
    } else {
        Some(y as usize * w + x as usize)
    }
}

fn morphological_close(img: &mut [u8], w: usize, h: usize) {
    let dil = dilate4(img, w, h);
    let closed = erode4(&dil, w, h);
    img.copy_from_slice(&closed);
}

fn dilate4(img: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = img.to_vec();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if img[y as usize * w + x as usize] != 0 {
                continue;
            }
            let n = [(0, -1), (0, 1), (-1, 0), (1, 0)];
            if n.iter().any(|&(dx, dy)| {
                idx(x + dx, y + dy, w, h).is_some_and(|i| img[i] != 0)
            }) {
                out[y as usize * w + x as usize] = 1;
            }
        }
    }
    out
}

fn erode4(img: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = img.to_vec();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if img[y as usize * w + x as usize] == 0 {
                continue;
            }
            let n = [(0, -1), (0, 1), (-1, 0), (1, 0)];
            if n.iter().any(|&(dx, dy)| {
                idx(x + dx, y + dy, w, h).is_none_or(|i| img[i] == 0)
            }) {
                out[y as usize * w + x as usize] = 0;
            }
        }
    }
    out
}

/// Zhang–Suen thinning. `img` is 0/1, row-major.
fn zhang_suen(img: &mut [u8], w: usize, h: usize) {
    loop {
        let mut marked = Vec::new();
        for y in 1..h as i32 - 1 {
            for x in 1..w as i32 - 1 {
                if img[y as usize * w + x as usize] == 0 {
                    continue;
                }
                if zhang_condition(img, w, x, y, true) {
                    marked.push((x, y));
                }
            }
        }
        for &(x, y) in &marked {
            img[y as usize * w + x as usize] = 0;
        }
        let n1 = marked.len();

        marked.clear();
        for y in 1..h as i32 - 1 {
            for x in 1..w as i32 - 1 {
                if img[y as usize * w + x as usize] == 0 {
                    continue;
                }
                if zhang_condition(img, w, x, y, false) {
                    marked.push((x, y));
                }
            }
        }
        for &(x, y) in &marked {
            img[y as usize * w + x as usize] = 0;
        }
        if n1 + marked.len() == 0 {
            break;
        }
    }
}

fn neighbors8(img: &[u8], w: usize, x: i32, y: i32) -> [u8; 8] {
    // p2..p9 clockwise from north
    let at = |dx, dy| img[(y + dy) as usize * w + (x + dx) as usize];
    [
        at(0, -1),
        at(1, -1),
        at(1, 0),
        at(1, 1),
        at(0, 1),
        at(-1, 1),
        at(-1, 0),
        at(-1, -1),
    ]
}

fn zhang_condition(img: &[u8], w: usize, x: i32, y: i32, first: bool) -> bool {
    let p = neighbors8(img, w, x, y);
    let b = p.iter().map(|&v| v as u32).sum::<u32>();
    if !(2..=6).contains(&b) {
        return false;
    }
    let mut a = 0u32;
    for i in 0..8 {
        if p[i] == 0 && p[(i + 1) % 8] != 0 {
            a += 1;
        }
    }
    if a != 1 {
        return false;
    }
    if first {
        p[0] * p[2] * p[4] == 0 && p[2] * p[4] * p[6] == 0
    } else {
        p[0] * p[2] * p[6] == 0 && p[0] * p[4] * p[6] == 0
    }
}

fn prune_spurs(img: &mut [u8], w: usize, h: usize, max_len: usize) {
    loop {
        let adj = adjacency(img, w, h);
        let mut removed = 0usize;
        for y in 0..h as i16 {
            for x in 0..w as i16 {
                let p = (x, y);
                let nbrs = match adj.get(&p) {
                    Some(n) if n.len() == 1 => n,
                    _ => continue,
                };
                let mut walk = vec![p];
                let mut prev = p;
                let mut curr = nbrs[0];
                loop {
                    walk.push(curr);
                    if walk.len() > max_len + 1 {
                        break;
                    }
                    let cn = adj.get(&curr).map(|n| n.as_slice()).unwrap_or(&[]);
                    if cn.len() != 2 {
                        break;
                    }
                    let next = if cn[0] == prev { cn[1] } else { cn[0] };
                    prev = curr;
                    curr = next;
                }
                let last = *walk.last().unwrap();
                let last_deg = adj.get(&last).map(|n| n.len()).unwrap_or(0);
                if walk.len() > 1 && walk.len() - 1 <= max_len && last_deg >= 3 {
                    for &q in &walk[..walk.len() - 1] {
                        img[q.1 as usize * w + q.0 as usize] = 0;
                        removed += 1;
                    }
                }
            }
        }
        if removed == 0 {
            break;
        }
    }
}

const N8: [(i16, i16); 8] = [
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];

fn adjacency(img: &[u8], w: usize, h: usize) -> HashMap<(i16, i16), Vec<(i16, i16)>> {
    let mut adj: HashMap<(i16, i16), Vec<(i16, i16)>> = HashMap::new();
    for y in 0..h as i16 {
        for x in 0..w as i16 {
            if img[y as usize * w + x as usize] == 0 {
                continue;
            }
            let mut nbrs = Vec::new();
            for (dx, dy) in N8 {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= w as i16 || ny >= h as i16 {
                    continue;
                }
                if img[ny as usize * w + nx as usize] != 0 {
                    nbrs.push((nx, ny));
                }
            }
            adj.insert((x, y), nbrs);
        }
    }
    adj
}

fn extract_paths(img: &[u8], w: usize, h: usize) -> Vec<Vec<(i16, i16)>> {
    let mut adj = adjacency(img, w, h);
    // Remove redundant diagonals from the graph (pixels stay).
    for y in 0..h as i16 {
        for x in 0..w as i16 {
            let Some(nbrs) = adj.get(&(x, y)).cloned() else {
                continue;
            };
            let has = |q: (i16, i16)| nbrs.contains(&q);
            let mut drop = Vec::new();
            if has((x + 1, y - 1)) && (has((x + 1, y)) || has((x, y - 1))) {
                drop.push((x + 1, y - 1));
            }
            if has((x + 1, y + 1)) && (has((x + 1, y)) || has((x, y + 1))) {
                drop.push((x + 1, y + 1));
            }
            if has((x - 1, y - 1)) && (has((x - 1, y)) || has((x, y - 1))) {
                drop.push((x - 1, y - 1));
            }
            if has((x - 1, y + 1)) && (has((x - 1, y)) || has((x, y + 1))) {
                drop.push((x - 1, y + 1));
            }
            if drop.is_empty() {
                continue;
            }
            adj.get_mut(&(x, y)).unwrap().retain(|n| !drop.contains(n));
            for d in drop {
                if let Some(list) = adj.get_mut(&d) {
                    list.retain(|n| *n != (x, y));
                }
            }
        }
    }

    let mut used: HashSet<((i16, i16), (i16, i16))> = HashSet::new();
    let edge = |a: (i16, i16), b: (i16, i16)| if a <= b { (a, b) } else { (b, a) };

    let walk = |start: (i16, i16),
                first: (i16, i16),
                adj: &HashMap<(i16, i16), Vec<(i16, i16)>>,
                used: &mut HashSet<((i16, i16), (i16, i16))>|
     -> Vec<(i16, i16)> {
        let mut path = vec![start];
        let mut prev = start;
        let mut curr = first;
        used.insert(edge(start, first));
        loop {
            path.push(curr);
            let nbrs = adj.get(&curr).map(|n| n.as_slice()).unwrap_or(&[]);
            if nbrs.len() != 2 {
                break;
            }
            let next = if nbrs[0] == prev { nbrs[1] } else { nbrs[0] };
            let e = edge(curr, next);
            if !used.insert(e) {
                break;
            }
            if next == start {
                path.push(next);
                break;
            }
            prev = curr;
            curr = next;
        }
        path
    };

    let mut paths = Vec::new();
    let mut pixels: Vec<(i16, i16)> = adj.keys().copied().collect();
    pixels.sort();

    // Endpoints first so we don't start in the middle of a stroke.
    for &p in &pixels {
        let nbrs = adj.get(&p).cloned().unwrap_or_default();
        if nbrs.len() != 1 {
            continue;
        }
        for n in nbrs {
            if !used.contains(&edge(p, n)) {
                paths.push(walk(p, n, &adj, &mut used));
            }
        }
    }
    // Leftover branches off junctions.
    for &p in &pixels {
        let nbrs = adj.get(&p).cloned().unwrap_or_default();
        if nbrs.len() < 3 {
            continue;
        }
        for n in nbrs {
            if !used.contains(&edge(p, n)) {
                paths.push(walk(p, n, &adj, &mut used));
            }
        }
    }
    // Closed loops with no endpoints (0, 8, 6, 9, D-like 4).
    for &p in &pixels {
        let nbrs = adj.get(&p).cloned().unwrap_or_default();
        for n in nbrs {
            if !used.contains(&edge(p, n)) {
                paths.push(walk(p, n, &adj, &mut used));
            }
        }
    }

    paths.retain(|p| p.len() >= 2);
    paths
}

fn order_paths(paths: &mut Vec<Vec<(i16, i16)>>) {
    if paths.is_empty() {
        return;
    }
    let mut remaining: Vec<Vec<(i16, i16)>> = std::mem::take(paths);
    let mut ordered = Vec::with_capacity(remaining.len());
    let mut pen: Option<(i16, i16)> = None;

    while !remaining.is_empty() {
        let mut best_i = 0usize;
        let mut best_rev = false;
        let mut best_key = (i32::MAX, i32::MAX, i32::MAX);

        for (i, path) in remaining.iter().enumerate() {
            let a = path[0];
            let b = path[path.len() - 1];
            let candidates = [(a, false), (b, true)];
            for (start, rev) in candidates {
                let key = if let Some(p) = pen {
                    let dx = start.0 as i32 - p.0 as i32;
                    let dy = start.1 as i32 - p.1 as i32;
                    (dx * dx + dy * dy, start.1 as i32, start.0 as i32)
                } else {
                    // First stroke: start at the topmost (then leftmost) end.
                    (start.1 as i32, start.0 as i32, 0)
                };
                if key < best_key {
                    best_key = key;
                    best_i = i;
                    best_rev = rev;
                }
            }
        }
        let mut path = remaining.swap_remove(best_i);
        if best_rev {
            path.reverse();
        }
        pen = path.last().copied();
        ordered.push(path);
    }
    *paths = ordered;
}

fn write_preview_bmp(path: &Path, src: &[u8], ink: &Ink) -> io::Result<()> {
    const MAG: usize = 4;
    let w = DST * MAG;
    let h = DST * MAG;
    let mut rgb = vec![0u8; w * h * 3];
    let gray = upsample_bilinear(src);
    for y in 0..DST {
        for x in 0..DST {
            let g = gray[y * DST + x];
            for dy in 0..MAG {
                for dx in 0..MAG {
                    let i = ((y * MAG + dy) * w + (x * MAG + dx)) * 3;
                    rgb[i] = g;
                    rgb[i + 1] = g;
                    rgb[i + 2] = g;
                }
            }
        }
    }
    for stroke in ink.strokes() {
        for pair in stroke.windows(2) {
            let x0 = (pair[0].x * MAG as f32).round() as i32;
            let y0 = (pair[0].y * MAG as f32).round() as i32;
            let x1 = (pair[1].x * MAG as f32).round() as i32;
            let y1 = (pair[1].y * MAG as f32).round() as i32;
            for (x, y) in bresenham(x0, y0, x1, y1) {
                for oy in -1..=1 {
                    for ox in -1..=1 {
                        let xx = x + ox;
                        let yy = y + oy;
                        if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                            let i = (yy as usize * w + xx as usize) * 3;
                            rgb[i] = 220;
                            rgb[i + 1] = 40;
                            rgb[i + 2] = 40;
                        }
                    }
                }
            }
        }
    }
    write_bmp(path, w as u32, h as u32, &rgb)
}

fn bresenham(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let mut pts = Vec::new();
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    loop {
        pts.push((x, y));
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
    pts
}

fn write_bmp(path: &Path, w: u32, h: u32, rgb: &[u8]) -> io::Result<()> {
    let row_stride = ((w * 3 + 3) / 4) * 4;
    let pixel_size = row_stride * h;
    let file_size = 54 + pixel_size;
    let mut f = File::create(path)?;
    f.write_all(b"BM")?;
    f.write_all(&file_size.to_le_bytes())?;
    f.write_all(&[0u8; 4])?;
    f.write_all(&54u32.to_le_bytes())?;
    f.write_all(&40u32.to_le_bytes())?;
    f.write_all(&w.to_le_bytes())?;
    f.write_all(&h.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&24u16.to_le_bytes())?;
    f.write_all(&0u32.to_le_bytes())?;
    f.write_all(&pixel_size.to_le_bytes())?;
    f.write_all(&[0u8; 16])?;
    let mut row = vec![0u8; row_stride as usize];
    for y in (0..h).rev() {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let o = (x * 3) as usize;
            // BMP is BGR.
            row[o] = rgb[i + 2];
            row[o + 1] = rgb[i + 1];
            row[o + 2] = rgb[i];
        }
        f.write_all(&row)?;
    }
    Ok(())
}
