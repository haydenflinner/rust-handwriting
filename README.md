# rust-handwriting

Online handwriting recognition in Rust, built on [Burn](https://burn.dev).

- **`hwr-ink`** — stroke/ink data model (points, pens, scratch-out gestures).
- **`hwr-model`** — the recognizer. An OnNeT-style BiLSTM+CTC over pen
  strokes, ported from the HuggingFace checkpoint and compiled natively
  by `burn-onnx` — see *Porting models from HuggingFace* below.
- **`hwr-app`** — a Bevy app for calibration, capture, and transcription,
  including a VLM backend (`oar-ocr-vl`) for whole-line OCR.

## Porting models from HuggingFace to Burn

The pipeline that works, end to end (`hwr-model/` + `scripts/`):

1. **Reimplement the model in PyTorch** — `scripts/onnet_torch.py`
   reproduces the upstream `pretrained-lstm` checkpoint layer for layer
   and loads its weights.
2. **Verify numerics before exporting** — `scripts/verify_onnet.py`
   and `scripts/ref_keras.py` check the port against real IAM-OnDB
   samples, so ONNX bugs are separable from porting bugs.
3. **Export ONNX with dynamic axes** — `scripts/export_onnx.py` emits
   `onnet_lstm.onnx` with a dynamic time axis; strokes arrive in any
   length.
4. **Codegen at build time** — `hwr-model/build.rs` runs `burn-onnx`,
   which compiles the graph into `src/onnet.rs` ahead of time. No
   Python, no ONNX runtime, no torch in the inference path — Burn with
   `wgpu`+`fusion` runs it on the GPU (Metal here).
5. **Fine-tune in torch, re-export** — `scripts/finetune_onnet.py`
   adapts the checkpoint on the app's own calibration corpus and writes
   `onnet_lstm_finetuned.onnx`; the crate tests
   (`tests/onnet_e2e.rs`) pin fixtures (ink → features → logits) so a
   bad re-export fails loudly.

The same shape — torch port → verified ONNX → `burn-onnx` codegen —
should transfer to other sequence models.

## VLMs for handwriting

We tested the general VLM path too (the app's `vlm` feature drives
`oar-ocr-vl`/Candle backends, HunyuanOCR included). For handwriting
specifically, **SmolVLM was the best of the VLMs we tried** — and at
500M it is small enough to download on demand rather than bundle. Its
weights need no conversion: `HuggingFaceTB/SmolVLM-500M-Instruct` is
stock safetensors that transformers/mlx-vlm load directly, and
`ggml-org` ships official GGUFs for llama.cpp. (It isn't wired into
`oar-ocr-vl` here — that backend family covers HunyuanOCR, PaddleOCR-VL,
GLM-OCR, etc. — so today SmolVLM runs through an external runner.)
For per-stroke online ink, though, the dedicated BiLSTM+CTC model above
remains the right tool: stroke-order information is simply unavailable
to an image model.
