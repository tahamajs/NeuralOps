Module neuralops.main
=====================
Efficient Attention Research Framework — Full Pipeline Runner
=============================================================
Single entry-point for:
  - Training    (fine-tune any HF model on GLUE / custom data)
  - Evaluation  (standalone eval of a saved checkpoint)
  - Compression (SVD / compressor-registry methods)
  - Sweep       (multi-seed, multi-attn-type grid run)
  - List        (print all registered attention / compressor keys)

Usage examples
--------------
# Basic training
python main.py train --attn_type full --task sst2 --epochs 3

# Compression then evaluate
python main.py compress --attn_type full --task sst2 --compressor svd
python main.py eval    --attn_type full --task sst2 --model_path results/...

# Grid sweep (all attention types x seeds)
python main.py sweep --attn_types full nha lsh --seeds 42 123 --task sst2

# List registry
python main.py list

Functions
---------

`build_parser() ‑> argparse.ArgumentParser`
:   Build the full CLI argument parser.

`load_data(cfg, tokenizer)`
:   Load, tokenise, and return a HuggingFace DatasetDict.

`main() ‑> None`
:   Main entry point — parse CLI args, import neuralops, run pipeline.

`run_compress(cfg, fw: dict, logger: logging.Logger) ‑> dict[str, typing.Any]`
:   Apply a registered compressor to the base model and report size reduction.

`run_eval(cfg, fw: dict, logger: logging.Logger, model_path: str) ‑> dict[str, typing.Any]`
:   Load a saved checkpoint and evaluate it on the validation split.

`run_list(fw: dict | None = None) ‑> None`
:   Print all registered components grouped by track.

`run_sweep(args, fw: dict, logger: logging.Logger) ‑> list[dict[str, typing.Any]]`
:   Run a grid sweep: attn_types × seeds, save aggregated results.

`run_train(cfg, fw: dict, logger: logging.Logger) ‑> dict[str, typing.Any]`
:   Full training pipeline: data → model → train → eval → save.

`setup_logger(log_dir: str, exp_name: str) ‑> logging.Logger`
:   Configure root logger to write to console + rotating file.

Classes
-------

`LiveHistoryCallback(out_dir)`
:   A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:
    
    Args:
        args ([`TrainingArguments`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            The model being trained.
        processing_class ([`PreTrainedTokenizer` or `BaseImageProcessor` or `ProcessorMixin` or `FeatureExtractionMixin`]):
            The processing class used for encoding the data. Can be a tokenizer, a processor, an image processor or a feature extractor.
        optimizer (`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for evaluation.
        metrics (`dict[str, float]`):
            The metrics computed by the last evaluation phase.
    
            Those are only accessible in the event `on_evaluate`.
        logs  (`dict[str, float]`):
            The values to log.
    
            Those are only accessible in the event `on_log`.
    
    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.
    
    The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformers.PrinterCallback`].
    
    Example:
    
    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_log(self, args, state, control, logs=None, **kwargs)`
    :   Event called after logging the last logs.