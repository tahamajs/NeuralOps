Module neuralops.utils.model_family
===================================
Architecture-family detection and common transformer model metadata.

Bug fixes (2026-09-25):
- Fix MF-01: Added "gpt2" (and gpt_neo, gpt_neox, llama, mistral, phi, falcon)
  to the causal family set. Previously ModelAdapter(family="gpt2") reported
  causal=False, causing causal masks to be dropped and recovery to fail on
  GPT-2. This was a critical correctness bug.
- Fix MF-02: Added GPT_FAMILIES and CAUSAL_FAMILIES frozensets to guarantee
  consistent behavior across causal, mask_rank, and position_embedding
  decisions.
- Fix MF-03: "gpt2" and other GPT variants now correctly report
  position_embedding="absolute" (learned position embeddings).
- Fix MF-04: mask_rank for all causal families is now 4.
- Fix MF-05: Added family alias normalization in ModelAdapter.__init__ so
  that any of {"gpt", "gpt2", "gpt_neo", "gpt_neox", "llama", "mistral",
  "phi", "falcon"} map to the canonical internal family "gpt", while
  preserving the raw family name in raw_family for logging.

Functions
---------

`detect_family(model_or_config: Any) ‑> str`
:   Detect the model family from a model or config object.

Classes
-------

`ModelAdapter(model_or_config: Any, family: str | None = None)`
:   Read-only architecture contract used by module wrappers.
    
    Fix MF-05: `family` is normalized to a canonical internal name (e.g. "gpt2"
    becomes "gpt"), while the original name is preserved in `raw_family` for
    logging. Downstream code that inspects `adapter.family` will always see a
    canonical value; code that needs the raw value should use `raw_family`.

    ### Methods

    `as_dict(self) ‑> dict[str, typing.Any]`
    :

    `projection(self, module: Any, kind: str) ‑> Any`
    :