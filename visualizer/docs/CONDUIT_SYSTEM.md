# 3D Dataflow Conduits & Inter-Module Wiring Engine

## 1. Conduits Overview

Modern transformer architectures are not isolated blocks; they are deeply connected networks of attention operators, multi-query head splits, gating mechanisms, and residual highways.

The visualizer computes code-accurate 3D cubic Bezier curves (`src/builders/conduit_builder.js`) that physically route token signals between operations.

```
       Layer Input Bus Port Ring (◎)
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
[Query Projection]   [Key/Value Split]
      │                     │
      └──────────┬──────────┘
                 ▼
          [Attention Core]
                 │
                 ▼
          [Output Dense]
                 │
                 ▼
       Residual Addition Node (+)  ◀──── [Bypass Highway Conduit]
```

---

## 2. Color-Coded Signal Bus Standards

| Signal Type | Color | Hex | Visual Role |
|---|---|---|---|
| **Input Stream** | Emerald Green | `#10b981` | Sequence entry from embedding lookup |
| **Baseline Feed** | Sky Blue | `#38bdf8` | Dense baseline forward connections |
| **Compressed Feed** | Mint Cyan | `#2dd4bf` | Low-rank / compressed forward connections |
| **GQA KV Bundle** | Amber Orange | `#f59e0b` | Shared Key/Value projections (Qwen) |
| **SwiGLU Branch** | Violet Purple | `#a855f7` | Bifurcated gate/up projection cables |
| **Residual Attn** | Deep Azure | `#0284c7` | Attention identity skip highway ($+X_{\text{in}}$) |
| **Residual FFN** | Royal Indigo | `#7c3aed` | Feed-forward identity skip highway ($+X_{\text{mid}}$) |
| **Output Head** | Crimson Rose | `#f43f5e` | LM Unembedding Softmax distribution |

---

## 3. Port Rings

At every layer entrance and exit, the engine instantiates a high-tech circular ingress/egress port ring (`THREE.RingGeometry`) indicating active tensor flow.
When a user clicks on any sublayer in the Deep Inspector or Component Pipeline, the associated conduits illuminate in bright emissive white to trace the signal pathway.
