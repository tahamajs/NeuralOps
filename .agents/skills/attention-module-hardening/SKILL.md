---
name: attention-module-hardening
description: Production-hardens a PyTorch attention or compression-engine module (under Codes/framework/attentions/ or Codes/framework/compression_engine/) so it works safely across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA-style models, then scaffolds its missing run_all_4_<module>.py benchmark driver, test_<module>_comprehensive.py test suite, and documentation stub. Use this whenever the user asks to "harden", "fix", "make production-ready", "complete the files for", or "validate" an attention/compressor module, names a module file directly (e.g. "harden eat.py"), or asks to process a list/queue of modules one by one with the same rules. Also trigger this for requests like "check which modules are missing run_all_4/test files" or "make sure this module handles config=None safely" — those are this skill's job even if phrased differently.
---

# Attention / Compressor Module Hardening

Use this any time you're asked to fix, harden, or complete file coverage for a module under
`Codes/framework/attentions/` or `Codes/framework/compression_engine/`. Apply it one module
at a time; never batch-edit multiple modules in a single unreviewed pass.

## Step 0 — Read before touching anything

1. `view` the target module's `.py` file completely.
2. `grep`/`view` its `run_all_4_<module>.py` (repo root) and `test_<module>_comprehensive.py`
   (`Codes/framework/tests/`) if they exist — don't recreate what's already correct.
3. Note every `@ATTENTION_REGISTRY.register(...)` / `@COMPRESSOR_REGISTRY.register(...)`
   name(s) — a module can have multiple aliases; the run_all_4 script and tests must use a
   name that actually resolves.

## Step 1 — The 9 hardening rules (apply every one that's relevant to this module's architecture)

1. **Safe config handling.** Every `getattr(config, ...)` must be guarded against
   `config=None`. Use three-branch logic everywhere hyperparameters are read:
   ```python
   if config is not None:
       self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
   elif comp_config is not None:
       self.lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
   else:
       self.lora_rank = kwargs.get("lora_rank", 16)
   ```
   Never call `getattr(config, "x", getattr(comp_config, ...))` unconditionally — if
   `config` is `None` that inner `getattr(comp_config, ...)` still evaluates fine (it has a
   default), but any *unconditional* `getattr(config, "x")` **without** a default will raise.
   Audit every config read in the file, not just the LoRA ones.

2. **Multi-architecture weight copy** in `copy_weights`/`_copy_pretrained`. Support, in order:
   - **DistilBERT**: flat `q_lin`/`k_lin`/`v_lin`/`out_lin`.
   - **RoBERTa/BERT nested**: `src.self.query/key/value` + outer `src.output.dense`. Set
     `self.is_nested_encoder = True` and bypass the output projection in `forward` when true
     (the outer module applies its own `output.dense` — applying yours too double-projects).
   - **GPT-2 fused Conv1D**: `src.c_attn` (split into Q/K/V thirds) and `src.c_proj`. Conv1D
     weight is stored `[in, out]` — transpose before splitting. If the layer is wrapped in
     `LoRALinear` (has a `base_layer`), extract from `base_layer`, not the wrapper; call
     `.merge()` first if present and not yet merged.
   - **Qwen/LLaMA/Mistral GQA**: `q_proj`/`k_proj`/`v_proj`/`o_proj`, with
     `num_key_value_heads` read from config and `kv_repeat = num_heads // num_kv_heads`.
   - After extraction, validate every copied projection with a Frobenius-norm check
     (`torch.linalg.norm(w.float()) > 1e-4`); raise `RuntimeError` if a mandatory weight is
     missing or near-zero — never silently proceed with garbage weights.

3. **LoRA adapters, frozen base.** Use a `LoRALinear` with `lora_A`/`lora_B` (standard naming
   so the recovery optimizer finds them). After `copy_weights`, call
   `target.freeze_base_weights()` on every projection so `requires_grad=False` on the base
   weight/bias and `True` on `lora_A`/`lora_B`.

4. **GQA support**: `num_kv_heads`, `kv_repeat`, and `repeat_kv(K, kv_repeat)` /
   `repeat_kv(V, kv_repeat)` applied *after* RoPE, before the attention matmul.

5. **RoPE**: accept `position_embeddings` via kwargs (apply to Q/K only, never V). If not
   provided, fall back to `self.rotary_emb` inherited from the source attention (store it in
   `copy_weights` via `getattr(src, "rotary_emb", None)`). Never depend on a nonexistent
   `self.adapter` in the fallback path.

6. **Causal masking**: detect `is_decoder` from `self.is_decoder or getattr(config, "is_decoder", False) or getattr(config, "model_type", "") in ("gpt2","qwen","qwen2","qwen3_5","llama")`
   and apply a lower-triangular mask *before* any top-k/sparsity/routing step (masking after
   is a causal-leakage bug). For linear/recurrent/SSM modules, ensure the recurrence itself is
   forward-only (no bidirectional scan on decoder paths).

7. **Nested encoder bypass** — see rule 2; don't forget the `forward` side of this, not just
   `copy_weights`.

8. **Safe injection.** Standard signature:
   ```python
   def inject_<module>(layer, model_config=None, comp_config=None, layer_idx=0,
                        total_layers=12, old_attn=None, **kwargs) -> Optional[nn.Module]:
   ```
   Never build a dummy config with hardcoded dims (e.g. `hidden_size=64, num_heads=4`) as a
   fallback — always try `old_attn.config` → `layer.config` → infer `hidden_size`/`num_heads`
   from `old_attn`'s actual projection shapes first. Wrap construction + weight copy in
   `try/except`; return `None` on any failure. Never let a randomly-initialized module get
   installed.

9. **Numerical stability**: compute attention scores in FP32, `torch.nan_to_num` after
   softmax, clamp MoE/expert outputs in FP16-sensitive paths.

## Step 2 — Create missing files (only the ones actually missing)

**A. `run_all_4_<module>.py`** (repo root) — copy the structure of an existing one (e.g.
`run_all_4_cluster.py`) rather than writing from scratch: argparse for `--models`,
`--recovery_steps`, `--batch_size`, `--precision`, `--lora_rank`, `--lora_alpha`; subprocess
into `run_quality_sweep.py`; parent checkpoints:
- DistilBERT: `results/unified_asvd/distilbert/finetuned_model.pt`
- RoBERTa: `results/roberta_sst2_lora_baseline/finetuned_model.pt`
- GPT-2: `results/gpt2_sst2_lora_baseline/finetuned_model.pt`
- Qwen3.5: `results/unified_asvd/qwen3_5/finetuned_model.pt`

Quality gate (status `PASS` and accuracy ≥ 56%) hands off to the `benchmark-and-publish`
skill — don't duplicate that logic here if it already lives in a shared runner.

**B. `test_<module>_comprehensive.py`** (`Codes/framework/tests/`) — must cover, at minimum:
- Injection succeeds on simulated DistilBERT/RoBERTa/GPT-2/Qwen clones.
- `config=None` with `comp_config` present still configures LoRA correctly.
- RoPE fallback fires when `position_embeddings` is absent but `source_attn.rotary_emb` exists.
- Frobenius-norm check raises on a zeroed reference weight.
- Causal and bidirectional forward passes return correct shapes and finite (non-NaN/Inf) values.
- Base weights end up frozen; `lora_A`/`lora_B` end up trainable.
- `copy_weights` failure path returns `None`, never a randomly-initialized module.

**C. Documentation stub** — if `Docs/content/attentions/<module>.md` (or `compressions/`)
doesn't exist, hand off to the `doc-code-alignment` skill to create it with the full
11-section structure rather than writing an ad-hoc doc here.

## Step 3 — Run and report

Run the new/updated test file locally: `pytest Codes/framework/tests/test_<module>_comprehensive.py -v`.
Fix failures before moving on — don't leave a red test and call the module "done."

Report a short status line per module: name, files created/fixed, test result (`X/X passed`),
and any open risk (e.g. "recurrent scan is O(L) sequential in pure PyTorch — will time out on
long sequences during recovery fine-tuning").

**Do not run the full 4-model GPU benchmark from this skill.** That's a separate, expensive
step — only invoke it when the user explicitly asks, at which point use the
`benchmark-and-publish` skill.

## When processing a queue of many modules

Go one at a time, in the order given. After each module, print the status line above before
starting the next — don't silently churn through the whole list and report only at the end;
long unattended runs are exactly where a subtle regression (e.g. a fix applied to module 12
but not module 40) goes unnoticed.
