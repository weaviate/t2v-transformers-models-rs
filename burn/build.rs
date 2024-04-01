use burn_import::onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("onnx/models/flan-t5-base/model.onnx")
        .out_dir("model")
        .run_from_script();
}