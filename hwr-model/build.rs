fn main() {
    // IAMhwr ONNET pretrained-lstm checkpoint, converted via
    // scripts/onnet_torch.py + scripts/export_onnx.py, then fine-tuned on
    // the app's calibration corpus + synthesized stroke data by
    // scripts/finetune_onnet.py (98 output classes; the .onnx here is a
    // committed build input).
    burn_onnx::ModelGen::new()
        .input("onnet_lstm_finetuned.onnx")
        .out_dir("onnet/")
        .run_from_script();
}
