Module neuralops.tests.test_results_saver
=========================================
Unit tests for ResultsSaver, metric exports, configuration archiving, and checkpoints.

Classes
-------

`TestResultsSaver(methodName='runTest')`
:   Test suite verifying ResultsSaver functionality.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `setUp(self)`
    :   Hook method for setting up the test fixture before exercising it.

    `tearDown(self)`
    :   Hook method for deconstructing the test fixture after testing it.

    `test_save_config(self)`
    :

    `test_save_metrics(self)`
    :

    `test_save_model_weights(self)`
    :