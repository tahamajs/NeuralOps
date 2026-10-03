Module neuralops.tests.test_threadsafe_rank_logging
===================================================
Unit tests for thread-safe QueueHandler logging and isolated rank-specific log files.

Classes
-------

`TestThreadSafeRankLogging(methodName='runTest')`
:   Test suite verifying QueueHandler thread-safety and rank_{LOCAL_RANK}.log isolation.
    
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

    `test_multithreaded_queue_logging_safety(self)`
    :

    `test_rank_specific_file_creation(self)`
    :