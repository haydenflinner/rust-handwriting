//! The default calibration prompt set: common programming symbols, numbers,
//! and words, grouped into categories and paginated into screen-sized
//! chunks. Every prompt is within the model's trained alphabet
//! (`hwr_model::decode::CHARS`), which happens to be all printable ASCII, so
//! this list doesn't need to be checked against it.

/// A single calibration screen: a titled grid of prompts.
pub struct Page {
    pub title: String,
    pub prompts: Vec<String>,
}

/// Cap on prompts per screen, so a big category (symbols, words) doesn't
/// produce an unreadably dense grid — it's split into multiple pages instead.
const MAX_PER_PAGE: usize = 24;

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

pub fn pages() -> Vec<Page> {
    let categories: &[(&str, &[&str])] = &[
        (
            "Digits",
            &["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        ),
        ("Lowercase Letters", LOWERCASE),
        ("Uppercase Letters", UPPERCASE),
        (
            "Numbers",
            &["10", "42", "100", "255", "1024", "2024", "3.14", "-1"],
        ),
        (
            "Symbols",
            &[
                "{", "}", "(", ")", "[", "]", "<", ">", "=", "==", "!=", "<=", ">=", "&&", "||",
                "!", "+", "-", "*", "/", "%", ";", ":", "::", ",", ".", "\"", "'", "`", "~", "@",
                "#", "$", "^", "&", "|", "\\", "_", "->", "=>", "+=", "-=", "//", "/*", "*/",
            ],
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
        ),
    ];

    let mut pages = Vec::new();
    for (name, prompts) in categories {
        let chunks: Vec<&[&str]> = prompts.chunks(MAX_PER_PAGE).collect();
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
            });
        }
    }
    pages
}
