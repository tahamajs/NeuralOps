Module neuralops.tests.test_track2_spectral_ssm_audit
=====================================================
Track 2 Comprehensive Audit & Dynamic Validation Suite.
Audits all Spectral, SSM, and Recurrent Attention modules:
- Forward pass correctness (output shape, no NaNs, no Infs)
- Backward pass gradient propagation (finite gradients on input and parameters)
- Variable sequence lengths and batch sizes
- Causal / padding attention mask compliance
- Numerical stability under extreme inputs
- Pretrained weight transfer compatibility

Classes
-------

`MockConfig(hidden_size=768, num_attention_heads=12, dropout=0.1)`
:   

`MockMHA(hidden_size=768)`
:   Mock pretrained MHA for weight transfer testing.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

`TestTrack2Modules(methodName='runTest')`
:   A class whose instances are single test cases.
    
    By default, the test code itself should be placed in a method named
    'runTest'.
    
    If the fixture may be used for many test cases, create as
    many test methods as are needed. When instantiating such a TestCase
    subclass, specify in the constructor arguments the name of the test method
    that the instance is to execute.
    
    Test authors should subclass TestCase for their own tests. Construction
    and deconstruction of the test's environment ('fixture') can be
    implemented by overriding the 'setUp' and 'tearDown' methods respectively.
    
    If it is necessary to override the __init__ method, the base class
    __init__ method must always be called. It is important that subclasses
    should not change the signature of their __init__ method, since instances
    of the classes are instantiated automatically by parts of the framework
    in order to be run.
    
    When subclassing TestCase, you can set these attributes:
    * failureException: determines which exception will be raised when
        the instance's assertion methods fail; test methods raising this
        exception will be deemed to have 'failed' rather than 'errored'.
    * longMessage: determines whether long messages (including repr of
        objects used in assert methods) will be printed on failure in *addition*
        to any explicit message passed.
    * maxDiff: sets the maximum length of a diff in failure messages
        by assert methods using difflib. It is looked up as an instance
        attribute so can be configured by individual tests if required.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self)`
    :   Hook method for setting up the test fixture before exercising it.

    `test_aegis_hybrid(self)`
    :

    `test_aegis_spectron_fp8(self)`
    :

    `test_aegis_v1(self)`
    :

    `test_aegis_v2(self)`
    :

    `test_aegis_v3(self)`
    :

    `test_aegis_v4(self)`
    :

    `test_aegis_v5(self)`
    :

    `test_aegis_v6(self)`
    :

    `test_galerkin_attention(self)`
    :

    `test_gated_delta_net(self)`
    :

    `test_gated_deltanet_block(self)`
    :

    `test_griffin_ssd(self)`
    :

    `test_halo_regla(self)`
    :

    `test_hgrn2_gated_deltanet(self)`
    :

    `test_hgrn2_layer(self)`
    :

    `test_hgrn2_outer(self)`
    :

    `test_kda_attention(self)`
    :

    `test_mamba(self)`
    :

    `test_mamba_shedder(self)`
    :

    `test_mamba_svd_flash(self)`
    :

    `test_regla_attention(self)`
    :

    `test_simple_ssm(self)`
    :

    `test_spectra_v2(self)`
    :

    `test_spectral_mamba(self)`
    :

    `test_svd_mamba(self)`
    :

    `test_upi_stuffed_mamba(self)`
    :

    `test_vspec_attention(self)`
    :