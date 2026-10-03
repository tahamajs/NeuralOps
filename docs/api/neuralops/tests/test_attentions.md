Module neuralops.tests.test_attentions
======================================

Classes
-------

`DummyConfig(hidden_size=64, num_heads=4, rank=16)`
:   

`TestAttentionMechanisms(methodName='runTest')`
:   Test suite verifying attention module registry, forward propagation, and output shapes.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self)`
    :   Hook method for setting up the test fixture before exercising it.

    `test_full_and_gqa_attention(self)`
    :

    `test_next_gen_hybrids(self)`
    :

    `test_registry_size(self)`
    :

    `test_svd_linear_attention(self)`
    :