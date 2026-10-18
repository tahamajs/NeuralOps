"""
CLI Serve Command

Launches high-throughput model serving engines and monitoring web services.

Author: Taha Majlesi
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
