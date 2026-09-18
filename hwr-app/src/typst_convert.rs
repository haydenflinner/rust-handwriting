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
        Ok(text) if !text.trim().is_empty() => text.trim().to_string(),
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
        Ok(body) => wrap_typst_math(body.trim()),
        Err(err) => format!("// mitex: {err}"),
    }
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
}
