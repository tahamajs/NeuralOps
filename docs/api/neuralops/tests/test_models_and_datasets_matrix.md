Module neuralops.tests.test_models_and_datasets_matrix
======================================================
Comprehensive Matrix Test Suite: Small Models (BERT, RoBERTa, DistilBERT, GPT-2)
across all Datasets (GLUE tasks, WikiText-2, C4, Multi-Choice) and SOTA Optimizers/Compressors.

Verifies 100% end-to-end compatibility with zero conflicts:
1. Model Architectures: BERT, RoBERTa, DistilBERT, GPT-2 causal LM.
2. Datasets: GLUE (sst2, mrpc, qnli, stsb, cola, rte, mnli), WikiText-2, C4, Multi-Choice.
3. Compression: Dynamic Entropy SVD, Res-SVD, AASVD, SliceGPT, Kronecker.
4. Optimization: Riemannian Stiefel Optimizer, Factorized Gradient Clipping, Mixed Precision AMP.

Classes
-------

`CompactBertForClassification(vocab_size: int = 1000, hidden_size: int = 64, num_layers: int = 2, num_heads: int = 4, num_labels: int = 2)`
:   Full BERT / RoBERTa classifier model for sequence classification benchmarks.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None, **kwargs: Any) ‑> Dict[str, torch.Tensor]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`CompactBertLayer(hidden_size: int = 64, num_heads: int = 4)`
:   Complete BERT / RoBERTa transformer layer.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`CompactBertSelfAttention(hidden_size: int = 64, num_heads: int = 4)`
:   BERT / RoBERTa self-attention layer with separate Q, K, V projections.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`CompactGPT2ForCausalLM(vocab_size: int = 1000, hidden_size: int = 64, num_layers: int = 2, num_heads: int = 4)`
:   Full GPT-2 causal language model with fused QKV and decay-gated attention options.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, labels: torch.Tensor | None = None, **kwargs: Any) ‑> Dict[str, torch.Tensor]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`TestModelsAndDatasetsMatrix()`
:   Comprehensive test matrix across models, GLUE datasets, language modeling, and compressors.

    ### Methods

    `test_bert_classification_on_glue_sst2(self) ‑> None`
    :   Verify BERT classification model trains on GLUE SST-2 dataset.

    `test_bert_res_svd_spectral_residual_lora(self) ‑> None`
    :   Verify Spectral Residual LoRA on BERT freezes base and updates adapter parameters.

    `test_distilbert_with_dynamic_entropy_svd(self) ‑> None`
    :   Verify DistilBERT compresses via Dynamic Entropy SVD and executes inference.

    `test_fused_qkv_slicing_on_all_attention_heads(self) ‑> None`
    :   Verify slice_fused_qkv and split_and_set_qkv_weights for standard MHA, GQA, and MQA.

    `test_gpt2_causal_lm_on_wikitext2_and_c4(self) ‑> None`
    :   Verify GPT-2 causal language model with decay linear attention trains on WikiText-2 and C4.

    `test_multi_choice_dataset_pipeline(self) ‑> None`
    :   Verify MultiChoiceDataset loading and tensor shaping for ARC/MMLU/HellaSwag tasks.

    `test_roberta_regression_on_glue_stsb(self) ‑> None`
    :   Verify RoBERTa regression model on continuous GLUE STS-B task.