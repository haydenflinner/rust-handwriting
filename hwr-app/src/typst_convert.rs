//! Convert Hunyuan's LaTeX to Typst using the same engine as the
//! [mitex](https://typst.app/universe/package/mitex/) Typst package
//! (`#mitex-convert(...)`).

/// Turn recognized (usually LaTeX) text into copy-pasteable Typst source.
pub fn latex_to_typst(raw: &str) -> String {
    let trimmed = raw.trim();
    if should_skip(trimmed) {
        return String::new();
    }

    if let Some(math) = extract_delimited_math(trimmed) {
        return convert_math_pretty(math);
    }

    let src = strip_fences(trimmed);
    if looks_like_latex_math(src) {
        return convert_math_pretty(src);
    }

    match mitex::convert_text(src, None) {
        Ok(text) if !text.trim().is_empty() => polish_typst(text.trim()),
        Ok(_) | Err(_) => convert_math_pretty(src),
    }
}

fn should_skip(s: &str) -> bool {
    s.is_empty() || s.starts_with("recognizing") || s.starts_with("(error:")
}

fn strip_fences(s: &str) -> &str {
    let s = s.trim();
    let Some(rest) = s.strip_prefix("```") else {
        return s;
    };
    let rest = rest
        .strip_prefix("latex")
        .or_else(|| rest.strip_prefix("tex"))
        .or_else(|| rest.strip_prefix("typst"))
        .unwrap_or(rest)
        .trim_start();
    rest.strip_suffix("```").map(str::trim).unwrap_or(s)
}

fn unwrap_pair<'a>(s: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let s = s.trim();
    s.strip_prefix(open)?
        .strip_suffix(close)
        .map(str::trim)
        .filter(|inner| !inner.is_empty())
}

fn extract_delimited_math(s: &str) -> Option<&str> {
    let s = s.trim();
    const PAIRS: &[(&str, &str)] = &[("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)")];
    for (open, close) in PAIRS {
        if let Some(inner) = unwrap_pair(s, open, close) {
            return Some(inner);
        }
        if let Some(start) = s.find(open) {
            let rest = &s[start + open.len()..];
            if let Some(end) = rest.find(close) {
                let inner = rest[..end].trim();
                if !inner.is_empty() {
                    return Some(inner);
                }
            }
        }
    }

    let t = s.trim();
    if t.starts_with('$') && t.ends_with('$') && t.len() > 2 {
        let inner = t[1..t.len() - 1].trim();
        if !inner.is_empty() && !inner.contains('$') {
            return Some(inner);
        }
    }
    None
}

fn looks_like_latex_math(s: &str) -> bool {
    s.contains('\\')
        || s.contains("begin{")
        || s.contains("frac")
        || s.contains("sum")
        || s.contains("int")
        || s.contains('^')
        || s.contains('_')
}

fn convert_math_pretty(src: &str) -> String {
    match mitex::convert_math(src, None) {
        Ok(body) => wrap_typst_math(&polish_typst(body.trim())),
        Err(err) => format!("// mitex: {err}"),
    }
}

/// Mitex emits package helpers (`mitexsqrt`, …) instead of vanilla Typst.
/// Strip the sqrt helper so the panel is copy-pasteable without `#import`.
fn polish_typst(src: &str) -> String {
    rewrite_mitexsqrt(src)
}

fn rewrite_mitexsqrt(src: &str) -> String {
    const NEEDLE: &str = "mitexsqrt(";
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(pos) = rest.find(NEEDLE) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + NEEDLE.len()..];
        match closing_paren(after) {
            Some((inner, consumed)) => {
                let inner = rewrite_mitexsqrt(inner);
                out.push_str(&sqrt_call_to_typst(&inner));
                rest = &after[consumed..];
            }
            None => {
                out.push_str(NEEDLE);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn sqrt_call_to_typst(inner: &str) -> String {
    let args = split_top_level_args(inner);
    match args.as_slice() {
        [only] => match unwrap_escaped_brackets(only.trim()) {
            Some(index) => format!("root({}, )", index.trim()),
            None => format!("sqrt({})", only.trim()),
        },
        [index, radicand] => {
            let index = unwrap_escaped_brackets(index.trim()).unwrap_or(index.trim());
            format!("root({}, {})", index.trim(), radicand.trim())
        }
        _ => format!("sqrt({})", inner.trim()),
    }
}

fn unwrap_escaped_brackets(s: &str) -> Option<&str> {
    s.strip_prefix(r"\[")?.strip_suffix(r"\]")
}

fn split_top_level_args(inner: &str) -> Vec<&str> {
    let mut args = Vec::new();
    let mut start = 0;
    let mut depth: u32 = 0;
    let mut escaped = false;
    for (i, c) in inner.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' => escaped = true,
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                args.push(&inner[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    args.push(&inner[start..]);
    args
}

fn closing_paren(s: &str) -> Option<(&str, usize)> {
    let mut depth: u32 = 1;
    let mut escaped = false;
    for (i, c) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' => escaped = true,
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&s[..i], i + c.len_utf8()));
                }
            }
            _ => {}
        }
    }
    None
}

fn wrap_typst_math(body: &str) -> String {
    if body.is_empty() {
        String::new()
    } else if body.contains('\n') {
        format!("$\n{body}\n$")
    } else {
        format!("${body}$")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_frac() {
        let typst = latex_to_typst(r"\frac{1}{2}");
        assert!(typst.contains("frac"), "{typst}");
        assert!(typst.starts_with('$') && typst.ends_with('$'), "{typst}");
    }

    #[test]
    fn strips_display_math() {
        let typst = latex_to_typst("$$\n\\alpha x\n$$");
        assert!(typst.contains("alpha"), "{typst}");
    }

    #[test]
    fn skips_status() {
        assert!(latex_to_typst("recognizing formula…").is_empty());
        assert!(latex_to_typst("").is_empty());
    }

    #[test]
    fn sqrt_becomes_typst_sqrt() {
        let typst = latex_to_typst(r"\sqrt{x}");
        assert!(!typst.contains("mitexsqrt"), "{typst}");
        assert!(typst.contains("sqrt("), "{typst}");
        assert!(typst.starts_with('$') && typst.ends_with('$'), "{typst}");
    }

    #[test]
    fn nth_root_becomes_typst_root() {
        let typst = latex_to_typst(r"\sqrt[3]{8}");
        assert!(!typst.contains("mitexsqrt"), "{typst}");
        assert!(typst.contains("root("), "{typst}");
        assert!(typst.contains("3"), "{typst}");
        assert!(typst.contains("8"), "{typst}");
    }

    #[test]
    fn nested_sqrt() {
        let typst = latex_to_typst(r"\sqrt{\sqrt{x}}");
        assert!(!typst.contains("mitexsqrt"), "{typst}");
        assert!(typst.contains("sqrt("), "{typst}");
    }
}
