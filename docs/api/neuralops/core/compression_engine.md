Module neuralops.core.compression_engine
========================================
Compression pipeline – orchestrates multiple compressors sequentially.

Functions
---------

`compress_model(model: torch.nn.modules.module.Module, config: dict[str, typing.Any], tokenizer=None) ‑> torch.nn.modules.module.Module`
:   Convenience function to compress a model with a configuration dict.

Classes
-------

`CompressionPipeline(model: torch.nn.modules.module.Module, config: dict[str, typing.Any], tokenizer=None)`
:   Apply a sequence of compression steps to a model.

    ### Methods

    `run(self) ‑> torch.nn.modules.module.Module`
    :   Execute the run operation and return results.