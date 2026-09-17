use hwr_ink::ink::Ink;
use hwr_model::{Greedy, Recognizer};

fn sample_ink() -> Ink {
    let mut ink = Ink::new();
    ink.push(0.0, 0.0, 0.0);
    ink.push(1.0, 1.0, 0.1);
    ink.push(2.0, 0.0, 0.2);
    ink.push(3.0, 1.0, 0.3);
    ink.pen_up();
    ink.push(0.0, 1.5, 0.4);
    ink.push(3.0, 1.5, 0.5);
    ink.pen_up();
    ink
}

#[test]
fn random_model_runs_end_to_end() {
    let recognizer = Recognizer::random().expect("build random recognizer");
    let ink = sample_ink();

    // With random weights the text is meaningless, but the whole pipeline —
    // HAT stroke+image encode, fused transformer, dense+softmax, CTC greedy
    // decode — must run without shape errors and produce *some* string.
    let text = recognizer
        .recognize_greedy(&ink)
        .expect("recognize should not error");

    eprintln!("random-weight decode: {text:?}");
}

#[test]
fn empty_ink_decodes_to_empty_string() {
    let recognizer = Recognizer::random().expect("build random recognizer");
    let text = recognizer
        .recognize(&Ink::new(), &Greedy)
        .expect("recognize should not error");
    assert_eq!(text, "");
}
