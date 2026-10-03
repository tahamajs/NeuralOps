Module neuralops.benchmarks.kd_quick_test
=========================================
Adaptive Knowledge Distillation (KD) Experiment on MambaBlock.

Evaluates whether KD can rescue an SSM-based module injected into GPT-2:
- Method A: Logits-only KD (0.9 * KL(T=2.0) + 0.1 * CE)
- Method B: Hidden-states only KD (0.9 * MSE_hidden + 0.1 * CE)
- Method C: Combined KD (0.3 * KL + 0.5 * MSE_hidden + 0.1 * MSE_attn + 0.1 * CE)
- Method D: SSM-Specialized KD (0.5 * KL(T=4.0) + 0.4 * Normalized/Cosine Hidden + 0.1 * CE)

Hardware Constraints: RTX 4070 Laptop GPU (8 GB VRAM).
Features:
- Teacher output preprocessing cache (5-10x speedup across iterations)
- Full VRAM release between sequential runs
- Automatic OOM fallback (halves seq_len to 128)
- Early stopping if PPL > 5000 (prevents wasted compute)
- Comprehensive CSV tracking, ASCII comparison table, and final JSON artifact

Functions
---------

`build_teacher_cache(teacher: torch.nn.modules.module.Module, tokenizer, train_chunks: torch.Tensor, cache_dir: pathlib.Path, methods: list[str], device: str, seq_len: int, include_attn: bool = False, max_train_texts: int = 1000) ‑> pathlib.Path`
:   Runs teacher forward on every chunk, saves outputs to cache_dir.
    Returns the path to the cache manifest.

`build_train_chunks(train_dataset, tokenizer, max_train_texts: int = 5000, seq_len: int = 256) ‑> torch.Tensor`
:   

`can_use_cache_for_method(cache: dict[str, typing.Any] | None, method_name: str, include_attn: bool = False) ‑> bool`
:   

`compute_attention_mse(student_attentions: tuple[torch.Tensor, ...] | None, teacher_attentions: tuple[torch.Tensor, ...] | None, device: str) ‑> torch.Tensor`
:   

`compute_hidden_mse(student_hidden_states: tuple[torch.Tensor, ...], teacher_hidden_states: tuple[torch.Tensor, ...], projection_adapter: neuralops.benchmarks.kd_quick_test.HiddenProjectionAdapter | None = None) ‑> torch.Tensor`
:   

`compute_kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) ‑> torch.Tensor`
:   

`compute_method_d_loss(student_out: Any, teacher_out: Any, input_ids: torch.Tensor, device: str, kl_temperature: float = 4.0, kl_weight: float = 0.5, hidden_weight: float = 0.4, ce_weight: float = 0.1, projection_adapter: neuralops.benchmarks.kd_quick_test.HiddenProjectionAdapter | None = None) ‑> torch.Tensor`
:   

`compute_perplexity(model: torch.nn.modules.module.Module, tokenizer, test_dataset, seq_len: int = 1024, stride: int = 512, max_eval_tokens: int = 15000, device: str = 'cuda') ‑> float`
:   

`get_cache_hash(model_path: str, tokenizer_name: str, seq_len: int, num_chunks: int, methods: list[str], include_attn: bool) ‑> str`
:   

`is_cache_valid(cache_dir: pathlib.Path, expected_chunks: int, expected_seq_len: int, methods: list[str], include_attn: bool = False) ‑> bool`
:   

`load_teacher_cache(cache_dir: pathlib.Path) ‑> dict[str, typing.Any]`
:   Loads cached outputs. Returns a dict:
      {
        "logits": Tensor(n_chunks, seq_len, vocab_size) fp16,
        "hidden_states": Tensor(n_layers, n_chunks, seq_len, hidden) fp16,
        "attentions": Tensor(n_layers, n_chunks, n_heads, seq_len, seq_len) fp16,
                        (or None if not cached),
        "manifest": {...}
      }

`main()`
:   

`make_run_dir(module_name: str, args: argparse.Namespace) ‑> pathlib.Path`
:   Create a unique run directory based on module name and config.

`print_comparison_table(results: list[dict[str, typing.Any]])`
:   

`run_kd_method(method_name: str, base_teacher: transformers.models.auto.modeling_auto.AutoModelForCausalLM, base_snapshot: transformers.models.auto.modeling_auto.AutoModelForCausalLM, tokenizer: transformers.models.auto.tokenization_auto.AutoTokenizer, train_chunks: torch.Tensor, test_dataset: Any, args: argparse.Namespace, teacher_cache: dict[str, typing.Any] | None = None, baseline_ppl: float = 21.71) ‑> dict[str, typing.Any]`
:   

`set_train_scope(model, train_scope='all')`
:   Configure trainable parameters.
    
    - 'all'                 : train everything
    - 'mamba'/'attention'/'module' : train ONLY parameters inside
                                    the attention submodule, EXCEPT:
                                      1. Original GPT-2 c_attn/c_proj weights
                                      2. SVD base weights (.U, .S, .V)
                                      3. FactorisedLinear base weights
                                         (.down.weight, .up.weight, .up.bias)
                                         when lora_A/lora_B adapters exist
                                    TRAINABLE: lora_A, lora_B, fusion_gate,
                                    prune_gate, SSM-specific params, etc.

`setup_logging(log_file: str | None = None)`
:   

`update_latest_link(module_name: str, run_dir: pathlib.Path)`
:   Record latest run path in results/06_adhoc_experiments/kd_runs/<module>/latest.txt.

`update_registry(module_name: str, run_dir: pathlib.Path, results: list[dict[str, typing.Any]], args: argparse.Namespace, baseline_ppl: float)`
:   Maintain a master JSON registry at results/06_adhoc_experiments/kd_runs/registry.json.

Classes
-------

`CachedTeacherOutput(logits: torch.Tensor | None = None, hidden_states: tuple[torch.Tensor, ...] | None = None, attentions: tuple[torch.Tensor, ...] | None = None)`
:   

`HiddenProjectionAdapter(hidden_dim: int, num_layers: int = 12)`
:   Lightweight learnable affine projection to bridge the architectural manifold gap
    between Transformer Attention (spatial dynamic routing) and Mamba/Recurrent SSM (temporal state-space).
    
    Instead of forcing strict isomorphic alignment (which over-restricts the student),
    this projection maps the student's hidden states to the teacher's manifold during KD
    and is safely discarded after distillation.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, layer_idx: int, student_h: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.