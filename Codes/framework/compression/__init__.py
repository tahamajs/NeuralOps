# framework/compression/__init__.py
"""
Model Compression Utilities for SVD-Lin Framework

This module provides various compression techniques for transformer models,
including SVD-based compression, adaptive methods, and hybrid approaches.

Available submodules:
- svd_based: Core SVD linear layer and rank selection
- advanced: Advanced zero-shot compression methods
- hybrid: Hybrid SVD encoder layers
- cur_cr_decomp: CUR/CR decomposition methods
- tt_decomposition: Tensor Train decomposition
- kronecker: Kronecker product methods
- low_rank_sparse: Combined low-rank and sparse methods
- quantization: Quantization techniques
- pruning: Pruning methods
- hashed: Hash-based compression
- moe: Mixture-of-Experts compression
- kv_cache: KV cache compression
- activation_aware: Activation-aware methods

Example usage:
    from framework.compression import SVDLinearLayer, AdvancedZeroShotCompressor
    
    # Apply SVD compression
    compressor = AdvancedZeroShotCompressor(method="standard_svd")
    compressed_model, stats, reduction = compressor.compress_model(model, "standard_svd")
"""

# ----------------------------------------------------------------------------
# Version information
# ----------------------------------------------------------------------------
__version__ = "2.0.0"
__author__ = "SVD-Lin Team"
__description__ = "Compression utilities for transformer models"

# ----------------------------------------------------------------------------
# Core exports from svd_based
# ----------------------------------------------------------------------------
from .svd_based import (
    SVDLinearLayer,
    get_adaptive_ranks,
    apply_adaptive_svd,
)

# ----------------------------------------------------------------------------
# Core exports from advanced
# ----------------------------------------------------------------------------
from .advanced import (
    AdvancedZeroShotCompressor,
    AdaptiveTransformerCompressor,
)

# ----------------------------------------------------------------------------
# Core exports from hybrid
# ----------------------------------------------------------------------------
from .hybrid import (
    SVDSingleHead,
    DirectSVDEncoderLayer,
)

# ----------------------------------------------------------------------------
# Conditional imports for optional submodules
# ----------------------------------------------------------------------------

# CUR/CR decomposition
try:
    from .cur_cr_decomp import (
        CURCompressor,
        CURLinear,
        CRCompressor,
        CURMoECompressor,
    )
    _HAS_CUR = True
except ImportError:
    _HAS_CUR = False
    CURCompressor = None
    CURLinear = None
    CRCompressor = None
    CURMoECompressor = None

# Tensor Train decomposition
try:
    from .tt_decomposition import (
        TTLinearLayer,
        TTCompressor,
    )
    _HAS_TT = True
except ImportError:
    _HAS_TT = False
    TTLinearLayer = None
    TTCompressor = None

# Kronecker methods
try:
    from .kronecker import (
        KroneckerLinear,
        KroneckerCompressor,
    )
    _HAS_KRONECKER = True
except ImportError:
    _HAS_KRONECKER = False
    KroneckerLinear = None
    KroneckerCompressor = None

# Low-rank + sparse
try:
    from .low_rank_sparse import (
        RPCACompressor,
        GreGoDecCompressor,
        SLiMCompressor,
        NMFCompressor,
    )
    _HAS_LOW_RANK_SPARSE = True
except ImportError:
    _HAS_LOW_RANK_SPARSE = False
    RPCACompressor = None
    GreGoDecCompressor = None
    SLiMCompressor = None
    NMFCompressor = None

# Quantization
try:
    from .quantization import (
        UniQLCompressor,
        MLoRQCompressor,
        KMeansQuantize,
        FAQQuantizer,
        HuffmanCoder,
    )
    _HAS_QUANT = True
except ImportError:
    _HAS_QUANT = False
    UniQLCompressor = None
    MLoRQCompressor = None
    KMeansQuantize = None
    FAQQuantizer = None
    HuffmanCoder = None

# Pruning
try:
    from .pruning import (
        SliceGPTCompressor,
        SparseGPTPruner,
        FLAPCompressor,
        OATSCompressor,
        MambaShedderCompressor,
        ARMORPruner,
        CoMeCompressor,
        ReplaceMeCompressor,
    )
    _HAS_PRUNING = True
except ImportError:
    _HAS_PRUNING = False
    SliceGPTCompressor = None
    SparseGPTPruner = None
    FLAPCompressor = None
    OATSCompressor = None
    MambaShedderCompressor = None
    ARMORPruner = None
    CoMeCompressor = None
    ReplaceMeCompressor = None

# Hash-based
try:
    from .hashed import (
        HashedLinear,
        ROASTLinear,
        HashedNetCompressor,
    )
    _HAS_HASHED = True
except ImportError:
    _HAS_HASHED = False
    HashedLinear = None
    ROASTLinear = None
    HashedNetCompressor = None

# MoE compression
try:
    from .moe import (
        MoESVDCompressor,
        CURMoECompressor as MoECURCompressor,
        ToMoECompressor,
    )
    _HAS_MOE = True
except ImportError:
    _HAS_MOE = False
    MoESVDCompressor = None
    MoECURCompressor = None
    ToMoECompressor = None

# KV cache compression
try:
    from .kv_cache import (
        DeltaKVCompressor,
        KVTCompressor,
    )
    _HAS_KV_CACHE = True
except ImportError:
    _HAS_KV_CACHE = False
    DeltaKVCompressor = None
    KVTCompressor = None

# Activation-aware methods
try:
    from .activation_aware import (
        FLATLLMCompressor,
        ActivationAwareSubspaceCompressor,
        COMPOTCompressor,
        LillamaCompressor,
    )
    _HAS_ACTIVATION = True
except ImportError:
    _HAS_ACTIVATION = False
    FLATLLMCompressor = None
    ActivationAwareSubspaceCompressor = None
    COMPOTCompressor = None
    LillamaCompressor = None

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def get_available_methods() -> dict:
    """
    Get all available compression methods with their status.
    
    Returns:
        Dictionary mapping method categories to lists of available methods
    """
    methods = {
        "core": ["SVDLinearLayer", "AdvancedZeroShotCompressor", "AdaptiveTransformerCompressor"],
        "svd_based": ["SVDLinearLayer", "get_adaptive_ranks", "apply_adaptive_svd"],
        "hybrid": ["SVDSingleHead", "DirectSVDEncoderLayer"],
    }
    
    if _HAS_CUR:
        methods["cur_cr"] = ["CURCompressor", "CURLinear", "CRCompressor"]
    if _HAS_TT:
        methods["tt_decomposition"] = ["TTLinearLayer", "TTCompressor"]
    if _HAS_KRONECKER:
        methods["kronecker"] = ["KroneckerLinear"]
    if _HAS_LOW_RANK_SPARSE:
        methods["low_rank_sparse"] = ["RPCACompressor", "GreGoDecCompressor", "SLiMCompressor", "NMFCompressor"]
    if _HAS_QUANT:
        methods["quantization"] = ["UniQLCompressor", "MLoRQCompressor", "KMeansQuantize", "FAQQuantizer"]
    if _HAS_PRUNING:
        methods["pruning"] = ["SliceGPTCompressor", "SparseGPTPruner", "FLAPCompressor", "OATSCompressor"]
    if _HAS_HASHED:
        methods["hashed"] = ["HashedLinear", "ROASTLinear"]
    if _HAS_MOE:
        methods["moe"] = ["MoESVDCompressor", "ToMoECompressor"]
    if _HAS_KV_CACHE:
        methods["kv_cache"] = ["DeltaKVCompressor", "KVTCompressor"]
    if _HAS_ACTIVATION:
        methods["activation_aware"] = ["FLATLLMCompressor", "COMPOTCompressor"]
    
    return methods


def print_available_methods():
    """Pretty print all available compression methods."""
    print("\n" + "=" * 60)
    print("📦 Available Compression Methods")
    print("=" * 60)
    
    methods = get_available_methods()
    for category, method_list in methods.items():
        print(f"\n🔹 {category.upper().replace('_', ' ')}:")
        for method in method_list:
            print(f"   - {method}")
    
    print("\n" + "=" * 60)


def get_compression_summary(model) -> dict:
    """
    Get a summary of compression applied to a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with compression statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Check for SVD layers
    svd_layers = 0
    for module in model.modules():
        if module.__class__.__name__ in ["SVDLinearLayer", "LoRALinear", "DoRALinear"]:
            svd_layers += 1
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "compressed_layers": svd_layers,
        "compression_ratio": 1 - (trainable_params / total_params) if total_params > 0 else 0,
    }

# ----------------------------------------------------------------------------
# Module exports
# ----------------------------------------------------------------------------

__all__ = [
    # Core
    "SVDLinearLayer",
    "get_adaptive_ranks",
    "apply_adaptive_svd",
    "AdvancedZeroShotCompressor",
    "AdaptiveTransformerCompressor",
    "SVDSingleHead",
    "DirectSVDEncoderLayer",
    
    # CUR/CR
    "CURCompressor",
    "CURLinear",
    "CRCompressor",
    "CURMoECompressor",
    
    # TT
    "TTLinearLayer",
    "TTCompressor",
    
    # Kronecker
    "KroneckerLinear",
    "KroneckerCompressor",
    
    # Low-rank sparse
    "RPCACompressor",
    "GreGoDecCompressor",
    "SLiMCompressor",
    "NMFCompressor",
    
    # Quantization
    "UniQLCompressor",
    "MLoRQCompressor",
    "KMeansQuantize",
    "FAQQuantizer",
    "HuffmanCoder",
    
    # Pruning
    "SliceGPTCompressor",
    "SparseGPTPruner",
    "FLAPCompressor",
    "OATSCompressor",
    "MambaShedderCompressor",
    "ARMORPruner",
    "CoMeCompressor",
    "ReplaceMeCompressor",
    
    # Hash-based
    "HashedLinear",
    "ROASTLinear",
    "HashedNetCompressor",
    
    # MoE
    "MoESVDCompressor",
    "ToMoECompressor",
    
    # KV cache
    "DeltaKVCompressor",
    "KVTCompressor",
    
    # Activation-aware
    "FLATLLMCompressor",
    "ActivationAwareSubspaceCompressor",
    "COMPOTCompressor",
    "LillamaCompressor",
    
    # Utilities
    "get_available_methods",
    "print_available_methods",
    "get_compression_summary",
]

# ----------------------------------------------------------------------------
# Module initialization
# ----------------------------------------------------------------------------

def _check_dependencies():
    """Check if optional dependencies are installed."""
    missing = []
    
    if not _HAS_CUR:
        missing.append("scikit-learn (for CUR/CR decomposition)")
    if not _HAS_TT:
        missing.append("torchtt (for Tensor Train decomposition)")
    if not _HAS_QUANT and 'sklearn' not in locals():
        missing.append("scikit-learn (for K-means quantization)")
    
    if missing:
        import warnings
        warnings.warn(
            f"Optional dependencies missing: {', '.join(missing)}. "
            "Some compression methods may not be available.",
            ImportWarning
        )

_check_dependencies()