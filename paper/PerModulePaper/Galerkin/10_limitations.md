<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Encoder Path Fails

Galerkin Attention **fails on encoders**:
- DistilBERT: -{{D_DROP}}%
- RoBERTa: -{{R_DROP}}% (chance level)

This is not a minor limitation; it is a fundamental failure of the method.

## 10.2 Decoder Path is a No-Op

For GPT-2, the module uses standard attention. There is no innovation on 
the decoder path.

## 10.3 No Parameter Reduction

Galerkin Attention **adds** parameters via LoRA. It is not a compression 
method.

## 10.4 Fixed Basis

The DCT basis is data-independent. This is the root cause of the encoder 
failure.

## 10.5 Task Coverage

Only SST-2. Behavior on generation, multi-class, or token-level tasks 
is untested.

## 10.6 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. No Qwen, LLaMA, or large models.

## 10.7 No Learned Basis Comparison

We did not compare against Linformer or Nyströmformer, which use learned 
or landmark-based bases.

## 10.8 Single Seed

No multi-seed runs.

## 10.9 No Theoretical Bound

We provide no formal approximation error bound.

## 10.10 Cross-Attention Not Supported

Passing `encoder_hidden_states` may fail. Not tested.

## 10.11 Failure Not Fully Diagnosed

While we attribute failure to the fixed basis, we do not prove this. Other 
factors (energy correction, QR orthonormalization, FP16 precision) may 
contribute.

## 10.12 Code Quality

- Duplicate `freeze_base_weights()` call in `copy_weights`
- No unit tests
- Some error paths log warnings but continue

## 10.13 What This Paper Is Not

- Not a compression method
- Not a speedup method
- Not a working attention replacement for encoders
```

---