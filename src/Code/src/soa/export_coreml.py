import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Convert model to Core ML.")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to input ONNX model.")
    parser.add_argument("--mlmodel_path", type=str, required=True, help="Path to output .mlmodel file.")
    args = parser.parse_args()

    # This repo's current coremltools build does NOT support ONNX conversion.
    # Fail fast with a clear message.
    raise RuntimeError(
        "This coremltools build does not support ONNX conversion. "
        "Please use a PyTorch -> CoreML export path instead, or install a "
        "coremltools version with the ONNX converter backend."
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ CoreML export failed: {e}")
        sys.exit(1)
