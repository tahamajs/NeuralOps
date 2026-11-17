try:
    from neuralops.core.triton_kernels.sliding_window import sliding_window_attention_kernel
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
