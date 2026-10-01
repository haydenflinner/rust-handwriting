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

We tested the general VLM path too. A browser bake-off
(Transformers.js + ONNX weights) pitted TrOCR-small, Florence-2-base
(-ft and base), LFM2.5-VL-450M, and SmolVLM-500M against handwritten
lines, equations, and code; **SmolVLM-500M was the best of the VLMs we
tried** and the only one that transcribed several samples verbatim —
and at 500M it is small enough to download on demand rather than
bundle.

Two catches for reproduction:

- HF never shipped an official ONNX build; the browser path runs a
  community export (`appleeatspi/pantrymax-smolvlm-500m-onnx-v2`,
  Idefics3 arch), fetched straight from HF at load — no local
  conversion needed. It only ships `q4`/`q4f16` variants; prefer `q4`,
  since `q4f16` can exceed WebGPU's storage-buffer limit.
- For non-browser use the stock HF safetensors work directly under
  transformers/mlx-vlm, and `ggml-org` ships official GGUFs for
  llama.cpp — but candle has no SmolVLM impl, so `oar-ocr-vl` (which
  covers HunyuanOCR, PaddleOCR-VL, GLM-OCR, etc.) can't run it.
For per-stroke online ink, though, the dedicated BiLSTM+CTC model above
remains the right tool: stroke-order information is simply unavailable
to an image model.
