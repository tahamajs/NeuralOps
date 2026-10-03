Module neuralops.tests.test_recurrent_ssm_attentions
====================================================
Unit tests for all 16 Recurrent, State Space & Linear DeltaNet Attention modules.

Verifies:
1. Clean instantiation from ATTENTION_REGISTRY.
2. Forward pass output shape (2, 128, 768) with no NaNs/Infs.
3. Backward pass gradient flow with non-null gradients on inputs and parameters.
4. Pretrained Multi-Head Attention weight transfer via copy_weights().
5. 2D and 4D attention mask handling.

Classes
-------

`DummyConfig(hidden_size: int = 768, num_attention_heads: int = 12, rank: int = 16)`
:   Mock config for 768-dim Transformer (DistilBERT / BERT).

`TestRecurrentSSMAttentions(methodName='runTest')`
:   Rigorous verification suite for all 16 recurrent & SSM attention modules.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self) ‑> None`
    :   Hook method for setting up the test fixture before exercising it.

    `test_01_mamba(self) ‑> None`
    :

    `test_02_mamba_svd_flash(self) ‑> None`
    :

    `test_03_gated_delta_net(self) ‑> None`
    :

    `test_04_gated_deltanet_block(self) ‑> None`
    :

    `test_05_hgrn2_layer(self) ‑> None`
    :

    `test_06_hgrn2_gated_deltanet(self) ‑> None`
    :

    `test_07_hgrn2_outer(self) ‑> None`
    :

    `test_08_griffin_ssd(self) ‑> None`
    :

    `test_09_hla_attention(self) ‑> None`
    :

    `test_10_kda_attention(self) ‑> None`
    :

    `test_11_simple_ssm(self) ‑> None`
    :

    `test_12_decay_linear_attention(self) ‑> None`
    :

    `test_13_gated_linear_attention(self) ‑> None`
    :

    `test_14_factorised_matrix_state(self) ‑> None`
    :

    `test_15_upi_stuffed_mamba(self) ‑> None`
    :