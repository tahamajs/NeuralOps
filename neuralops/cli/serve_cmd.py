# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Serve Command

Launches high-throughput model serving engines and monitoring web services.
"""

def cmd_serve(args):
    try:
        import uvicorn
        from neuralops.api.app import app
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except ImportError:
        print("Install: pip install uvicorn fastapi")
        return 1
    except Exception as e:
        print(f"API server not available: {e}")
        return 1
    return 0
