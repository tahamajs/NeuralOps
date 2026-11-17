try:
    from neuralops.ops.triton.sliding_window import sliding_window_attention_kernel
    print("Direct import successful")
except ImportError as e:
    print(f"Direct import failed: {e}")
except Exception as e:
    print(f"Direct import other error: {e}")
