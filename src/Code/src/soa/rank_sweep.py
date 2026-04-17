import subprocess
import sys
import argparse

# List of ranks to test
RANKS = [32, 48, 64, 96, 128]

def run(skip_coreml=False):
    for r in RANKS:
        onnx_path = f"soa_bert_rank_{r}.onnx"
        ml_path = f"soa_bert_rank_{r}.mlmodel"

        print(f"\n{'='*40}")
        print(f"Processing Rank: {r}")
        print(f"{'='*40}")

        # 1. Convert to ONNX
        try:
            subprocess.run(
                ["python", "-m", "soa.export_onnx", "--rank", str(r), "--onnx_path", onnx_path],
                check=True
            )
            print(f"✅ ONNX exported: {onnx_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to export ONNX for rank {r}: {e}")
            continue

        # 2. Convert to CoreML (Optional)
        if not skip_coreml:
            try:
                subprocess.run(
                    ["python", "-m", "soa.export_coreml", "--onnx_path", onnx_path, "--mlmodel_path", ml_path],
                    check=True
                )
                print(f"✅ CoreML exported: {ml_path}")
            except FileNotFoundError:
                print(f"⚠️ Module 'soa.export_coreml' not found. Skipping CoreML export.")
            except subprocess.CalledProcessError:
                print(f"⚠️ CoreML export failed for rank {r} (likely unsupported coremltools). Skipping.")
            except Exception as e:
                print(f"⚠️ Unexpected error during CoreML export for rank {r}: {e}")
        else:
            print("ℹ️ Skipping CoreML export as requested.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rank sweep for SOA model.")
    parser.add_argument("--skip_coreml", action="store_true", help="Skip the CoreML conversion step.")
    args = parser.parse_args()

    print("🚀 Starting Rank Sweep Process...")
    run(skip_coreml=args.skip_coreml)
    print("\n🏁 Rank Sweep Process Completed.")
