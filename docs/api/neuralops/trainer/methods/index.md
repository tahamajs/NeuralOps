Module neuralops.trainer.methods
================================
Training method plugins: LoRA, DoRA, DisLoRA, Fisher, GeLoRA, LLRD, Mixout, R‑Drop, R3F.
All methods are registered via the METHOD_REGISTRY and can be applied by name.

Sub-modules
-----------
* neuralops.trainer.methods.fisher
* neuralops.trainer.methods.gelora
* neuralops.trainer.methods.llrd
* neuralops.trainer.methods.lora
* neuralops.trainer.methods.mixout
* neuralops.trainer.methods.rdrop_r3f
* neuralops.trainer.methods.registry

Functions
---------

`apply_dislora(model, config) ‑> None`
:   Apply dislora.

`apply_dora(model, config) ‑> None`
:   Apply dora.

`apply_fisher_gelora(model, config) ‑> None`
:   Apply fisher gelora.

`apply_fisher_lora(model, config) ‑> None`
:   Apply fisher lora.

`apply_llrd(model, config) ‑> None`
:   Apply llrd.

`apply_lora(model, config) ‑> None`
:   Apply lora.

`apply_mixout(model, config) ‑> None`
:   Apply mixout.

`apply_r3f(model, config) ‑> None`
:   Apply r3f.

`apply_rdrop(model, config) ‑> None`
:   Apply rdrop.

`available_methods() ‑> Any`
:   Return list of all registered method names.

`collect_hidden_states(model, dataloader, num_batches=10, max_samples=800) ‑> Any`
:   Collect hidden states.

`compute_fisher_importance(model, dataset, tokenizer, task_keys, config, target_substrings=None) ‑> Any`
:   Compute and return fisher importance.

`compute_gelora_base_ranks(model, dataloader, config) ‑> Any`
:   Compute and return gelora base ranks.

`estimate_intrinsic_dimension_pytorch(hidden_states, k=5) ‑> Any`
:   Estimate intrinsic dimension pytorch.

`get_llrd_param_groups(model, base_lr, weight_decay, decay) ‑> Any`
:   Return the llrd param groups.

`get_method(name: str) ‑> Any`
:   Return the method function by name, or None if not found.

`r3f_forward(model, input_ids, attention_mask, token_type_ids, labels, epsilon, is_regression) ‑> Any`
:   R3f forward.

`rdrop_loss(logits1, logits2, labels, is_regression, alpha) ‑> Any`
:   Rdrop loss.

`register_method(name: str) ‑> Any`
:   Decorator to register a method callable.

Classes
-------

`DisLoRALinear(linear: torch.nn.modules.linear.Linear, r: int, alpha: float, dropout: float, beta: float = 0.5, capacity_factor: float = 1.0)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Any`
    :   Forward pass — compute module output from inputs.

    `ortho_loss(self) ‑> Any`
    :   Ortho loss.

`DoRALinear(linear: torch.nn.modules.linear.Linear, r: int, alpha: float, dropout: float)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Any`
    :   Forward pass — compute module output from inputs.

`LoRALinear(linear: torch.nn.modules.linear.Linear, r: int, alpha: float, dropout: float)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Any`
    :   Forward pass — compute module output from inputs.

`MethodTrainer(method, rdrop_alpha, r3f_epsilon, is_regression, *args, **kwargs)`
:   Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.
    
    Args:
        model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.
    
            <Tip>
    
            [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
            your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
            models.
    
            </Tip>
    
        args ([`TrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator (`DataCollator`, *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`default_data_collator`] if no `processing_class` is provided, an instance of
            [`DataCollatorWithPadding`] otherwise if the processing_class is a feature extractor or tokenizer.
        train_dataset (`torch.utils.data.Dataset` | `torch.utils.data.IterableDataset` | `datasets.Dataset`, *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.
    
            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (`torch.utils.data.Dataset` | dict[str, `torch.utils.data.Dataset`] | `datasets.Dataset`, *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        model_init (`Callable[[], PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
            from a new instance of the model as given by this function.
    
            The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to
            be able to choose different architectures according to hyperparameters (such as layer count, sizes of
            inner layers, dropout probabilities etc).
        compute_loss_func (`Callable`, *optional*):
            A function that accepts the raw model outputs, labels, and the number of items in the entire accumulated
            batch (batch_size * gradient_accumulation_steps) and returns the loss. For example, see the default [loss function](https://github.com/huggingface/transformers/blob/052e652d6d53c2b26ffde87e039b723949a53493/src/transformers/trainer.py#L3618) used by [`Trainer`].
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics
        callbacks (List of [`TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).
    
            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`tuple[Type[torch.optim.Optimizer], dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use.
            Overrides `optim` and `optim_args` in `args`. Incompatible with the `optimizers` argument.
    
            Unlike `optimizers`, this argument avoids the need to place model parameters on the correct devices before initializing the Trainer.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.
    
            Note that the labels (second parameter) will be `None` if the dataset does not have them.
    
    Important attributes:
    
        - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
          subclass.
        - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
          original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
          the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
          model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
        - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
          data parallelism, this means some of the model layers are split on different GPUs).
        - **place_model_on_device** -- Whether or not to automatically place the model on the device. Defaults to
          `True` unless model parallel, DeepSpeed, FSDP, full fp16/bf16 eval, or SageMaker MP is active. Can be
          overridden by subclassing `TrainingArguments` and overriding the `place_model_on_device` property.
        - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
          in `train`)

    ### Ancestors (in MRO)

    * transformers.trainer.Trainer

    ### Methods

    `compute_loss(self, model, inputs, return_outputs=False, **kwargs) ‑> Any`
    :   Compute and return loss.

`MixoutLinear(linear: torch.nn.modules.linear.Linear, p: float)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
    
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.
    
    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x) ‑> Any`
    :   Forward pass — compute module output from inputs.