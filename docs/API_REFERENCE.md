# NeuralOps v1.1.0 API Reference

## Core Registries

- **ATTENTION_REGISTRY**: 145 registered modules.
- **COMPRESSOR_REGISTRY**: 73 registered modules.

## Key Modules

### Inference Engine

- **`NeuralOpsInferenceEngine`**: Full-featured greedy/top-k/top-p/repetition-penalty generation runtime.
- **`stream_generate`**: Yields generated tokens one-by-one.
- **`speculative_generate`**: Real speculative decoding with acceptance/rejection sampling.
- **`beam_search_generate`**: Production beam search decoder with length normalization.
- **`GenerationRequest`**: Represents a serving generation request with its generation state.
- **`ContinuousBatcher`**: Continuous dynamic batch scheduler admitting requests on the fly.

### GPTQ Quantizer

- **`GPTQQuantizer`**: Second-order Hessian-guided post-training weight quantization with column error propagation.
- **`quantize_model_gptq`**: Quantizes all nn.Linear layers in model using GPTQ.

### AWQ Quantizer

- **`AWQQuantizer`**: Protects salient weights by observing activation magnitudes and re-scaling channels.
- **`quantize_model_awq`**: Applies AWQ quantization across all linear layers in model.

### SmoothQuant

- **`smoothquant_layer`**: Applies SmoothQuant to a linear layer.
- **`smoothquant_model`**: Applies SmoothQuant to all nn.Linear layers in model and returns scale dict.

### Wanda Pruner

- **`wanda_prune_layer`**: Prunes linear layer weights based on the Wanda importance score S_ij = |W_ij| * ||X_j||_2.
- **`wanda_prune_model`**: Applies Wanda pruning to all linear layers of model.

