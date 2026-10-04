try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = type(
        "triton",
        (),
        {
            "jit": lambda f: f,
            "heuristics": lambda *a, **kw: lambda f: f,
            "autotune": lambda *a, **kw: lambda f: f,
        },
    )
    tl = None
from packaging import version

if HAS_TRITON and hasattr(triton, "__version__"):
    TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")
else:
    TRITON3 = False


if TRITON3:

    @triton.jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)

else:

    @triton.jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt))
