Module neuralops.config.method_config
=====================================

Classes
-------

`DisLoRAConfig(beta: float = 0.5, capacity_factor: float = 1.0)`
:   DisLoRAConfig(beta: float = 0.5, capacity_factor: float = 1.0)

    ### Instance variables

    `beta: float`
    :   The type of the None singleton.

    `capacity_factor: float`
    :   The type of the None singleton.

`FisherConfig(num_batches: int = 20, sample_size: int = 200, scale_max: float = 2.0)`
:   FisherConfig(num_batches: int = 20, sample_size: int = 200, scale_max: float = 2.0)

    ### Instance variables

    `num_batches: int`
    :   The type of the None singleton.

    `sample_size: int`
    :   The type of the None singleton.

    `scale_max: float`
    :   The type of the None singleton.

`GeLoRAConfig(margin: float = 1.0, min_rank: int = 2, max_rank: int = 32)`
:   GeLoRAConfig(margin: float = 1.0, min_rank: int = 2, max_rank: int = 32)

    ### Instance variables

    `margin: float`
    :   The type of the None singleton.

    `max_rank: int`
    :   The type of the None singleton.

    `min_rank: int`
    :   The type of the None singleton.

`LLRDConfig(decay: float = 0.95)`
:   LLRDConfig(decay: float = 0.95)

    ### Instance variables

    `decay: float`
    :   The type of the None singleton.

`LoRAConfig(r: int = 8, alpha: int = 16, dropout: float = 0.1)`
:   LoRAConfig(r: int = 8, alpha: int = 16, dropout: float = 0.1)

    ### Instance variables

    `alpha: int`
    :   The type of the None singleton.

    `dropout: float`
    :   The type of the None singleton.

    `r: int`
    :   The type of the None singleton.

`MethodConfig(method: str = 'baseline', lora: neuralops.config.method_config.LoRAConfig = <factory>, rdrop: neuralops.config.method_config.RDropConfig = <factory>, r3f: neuralops.config.method_config.R3FConfig = <factory>, fisher: neuralops.config.method_config.FisherConfig = <factory>, gelora: neuralops.config.method_config.GeLoRAConfig = <factory>, llrd: neuralops.config.method_config.LLRDConfig = <factory>, mixout: neuralops.config.method_config.MixoutConfig = <factory>, dislora: neuralops.config.method_config.DisLoRAConfig = <factory>)`
:   MethodConfig(method: str = 'baseline', lora: neuralops.config.method_config.LoRAConfig = <factory>, rdrop: neuralops.config.method_config.RDropConfig = <factory>, r3f: neuralops.config.method_config.R3FConfig = <factory>, fisher: neuralops.config.method_config.FisherConfig = <factory>, gelora: neuralops.config.method_config.GeLoRAConfig = <factory>, llrd: neuralops.config.method_config.LLRDConfig = <factory>, mixout: neuralops.config.method_config.MixoutConfig = <factory>, dislora: neuralops.config.method_config.DisLoRAConfig = <factory>)

    ### Instance variables

    `dislora: neuralops.config.method_config.DisLoRAConfig`
    :   The type of the None singleton.

    `fisher: neuralops.config.method_config.FisherConfig`
    :   The type of the None singleton.

    `gelora: neuralops.config.method_config.GeLoRAConfig`
    :   The type of the None singleton.

    `llrd: neuralops.config.method_config.LLRDConfig`
    :   The type of the None singleton.

    `lora: neuralops.config.method_config.LoRAConfig`
    :   The type of the None singleton.

    `method: str`
    :   The type of the None singleton.

    `mixout: neuralops.config.method_config.MixoutConfig`
    :   The type of the None singleton.

    `r3f: neuralops.config.method_config.R3FConfig`
    :   The type of the None singleton.

    `rdrop: neuralops.config.method_config.RDropConfig`
    :   The type of the None singleton.

`MixoutConfig(p: float = 0.9)`
:   MixoutConfig(p: float = 0.9)

    ### Instance variables

    `p: float`
    :   The type of the None singleton.

`R3FConfig(epsilon: float = 1e-05)`
:   R3FConfig(epsilon: float = 1e-05)

    ### Instance variables

    `epsilon: float`
    :   The type of the None singleton.

`RDropConfig(alpha: float = 5.0)`
:   RDropConfig(alpha: float = 5.0)

    ### Instance variables

    `alpha: float`
    :   The type of the None singleton.