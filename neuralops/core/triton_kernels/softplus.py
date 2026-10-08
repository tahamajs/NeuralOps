from packaging import version
from neuralops.core.triton_kernels._common import HAS_TRITON, tl, triton


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
