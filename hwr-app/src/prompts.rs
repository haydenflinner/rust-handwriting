//! The default calibration prompt set: common programming symbols, numbers,
//! and words, grouped into categories and paginated into screen-sized
//! chunks. Every prompt is within the model's trained alphabet
//! (`hwr_model::onnet::CHARS`), which happens to be all printable ASCII, so
//! this list doesn't need to be checked against it.

/// A single calibration screen: a titled grid of prompts.
pub struct Page {
    pub title: String,
    pub prompts: Vec<String>,
    /// Full-width cells for whole lines of code, instead of the small
    /// per-glyph grid boxes.
    pub line_mode: bool,
}

/// Cap on prompts per screen, so a big category (symbols, words) doesn't
/// produce an unreadably dense grid — it's split into multiple pages instead.
const MAX_PER_PAGE: usize = 24;
/// Fewer per screen in line mode — each prompt gets a full-width cell tall
/// enough to handwrite a line of code.
const LINES_PER_PAGE: usize = 7;

/// Lowercase and uppercase letters, one at a time. Recorded as isolated
/// single-character ink samples, these double as a "font": individual
/// letters can be spliced together (positioned/spaced/timed, with pen-lifts
/// between them) into words that were never written by hand, drawing on
/// whatever vocabulary the training corpus needs — see `hwr-model`'s
/// planned splicing tool.
const LOWERCASE: &[&str] = &[
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z",
];
const UPPERCASE: &[&str] = &[
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];

/// Whole lines of code, one prompt each — line-level samples are the input
/// distribution the model was actually trained on (context, spacing,
/// multi-char segmentation), so these are the highest-value samples to
/// collect. Syntax follows mimas (dsa/host/web/src/mimas.ts): keywords
/// self/let/return/raise/absolve/if/else/null/struct/pact/impl/const/pub/
/// module/break/collect/fn/true/false/for/in/match/continue/enum/loop/while/
/// use, contextual where/examples/is, types int/float/str/bool, // comments,
/// f"/r#" string prefixes.
const CODE_DECLARATIONS: &[&str] = &[
    "fn area(w: float, h: float) -> float { w * h }",
    "let mut count = 0;",
    "const MAX_RETRIES = 5;",
    "pub struct Token { kind, text }",
    "fn is_digit(c) -> bool { c >= '0' && c <= '9' }",
    "let (a, b) = (b, a + b);",
    "impl Token { fn new() -> Self }",
];
const CODE_EXPRESSIONS: &[&str] = &[
    "if x % 2 == 0 { parity = \"even\"; }",
    "let ok = a && b || !c;",
    "acc += dx * dy - 0.5;",
    "mask = bits << 4 | 0x0F;",
    "while i < n - 1 { i += 1; }",
    "d = (p.x - q.x).abs() + (p.y - q.y).abs();",
    "x = x * 2 + 1; y /= 3;",
];
const CODE_DATA: &[&str] = &[
    "let xs = [0.25, 1.5, 2.75, 42.0];",
    "nums.push(1024 * 3 + 17);",
    "let t = 1_000_000.0 / 60.0;",
    "let range = 0..=255;",
    "score = 100 * hits / tries;",
    "let id = 0xDEAD_BEEF;",
    "let msg = f\"answer is {x}\";",
];
const CODE_CONTROL: &[&str] = &[
    "match ch { '0'..='9' => true, _ => false }",
    "for (i, line) in text.lines() { collect(line); }",
    "if result is Ok(v) { out.push(v); }",
    "example: parse(\"1 + 2\") == 3;",
    "where x > 0 && y < 100;",
    "raise \"bad char at {pos}\";",
    "// TODO: handle the empty case",
];

pub fn pages() -> Vec<Page> {
    let categories: &[(&str, &[&str], bool)] = &[
        (
            "Digits",
            &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            false,
        ),
        ("Lowercase Letters", LOWERCASE, false),
        ("Uppercase Letters", UPPERCASE, false),
        (
            "Numbers",
            &["10", "42", "100", "255", "1024", "2024", "3.14", "-1"],
            false,
        ),
        (
            "Symbols",
            &[
                "{", "}", "(", ")", "[", "]", "<", ">", "=", "==", "!=", "<=", ">=", "&&", "||",
                "!", "+", "-", "*", "/", "%", ";", ":", "::", ",", ".", "\"", "'", "`", "~", "@",
                "#", "$", "^", "&", "|", "\\", "_", "->", "=>", "+=", "-=", "//", "/*", "*/",
            ],
            false,
        ),
        (
            "Words",
            &[
                "if", "else", "for", "while", "return", "function", "class", "import", "export",
                "true", "false", "null", "nil", "none", "let", "const", "var", "def", "print",
                "self", "this", "public", "private", "static", "void", "int", "string", "bool",
                "float", "struct", "enum", "match", "case", "switch", "break", "continue", "fn",
                "impl", "pub", "use", "mod", "trait", "async", "await", "try", "catch", "throw",
                "new", "delete",
            ],
            false,
        ),
        ("Code: declarations", CODE_DECLARATIONS, true),
        ("Code: expressions", CODE_EXPRESSIONS, true),
        ("Code: numbers & data", CODE_DATA, true),
        ("Code: control flow", CODE_CONTROL, true),
    ];

    let mut pages = Vec::new();
    for (name, prompts, line_mode) in categories {
        let per_page = if *line_mode { LINES_PER_PAGE } else { MAX_PER_PAGE };
        let chunks: Vec<&[&str]> = prompts.chunks(per_page).collect();
        let total = chunks.len();
        for (i, chunk) in chunks.into_iter().enumerate() {
            let title = if total > 1 {
                format!("{name} ({}/{total})", i + 1)
            } else {
                name.to_string()
            };
            pages.push(Page {
                title,
                prompts: chunk.iter().map(|s| s.to_string()).collect(),
                line_mode: *line_mode,
            });
        }
    }
    pages
}
