from typing import Any
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False

    class _DummyTriton:
        @staticmethod
        def jit(*args, **kwargs):
            def decorator(fn):
                return fn

            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator

        @staticmethod
        def autotune(*args, **kwargs):
            def decorator(fn):
                return fn

            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator

        @staticmethod
        def Config(*args, **kwargs):
            return {}

        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

        @staticmethod
        def next_power_of_2(n):
            return 1 if n == 0 else 2 ** (int(n) - 1).bit_length()

    class _DummyTL:
        constexpr = Any

    triton = _DummyTriton
    tl = _DummyTL

