Module neuralops.tests.test_datasets
====================================
Unit tests for benchmark dataset registry, loaders, and synthetic generation.

Classes
-------

`TestBenchmarkDatasets(methodName='runTest')`
:   Test suite verifying dataset loading, batch collation, and synthetic fallback.
    
    Create an instance of the class that will use the named test
    method when executed. Raises a ValueError if the instance does
    not have a method with the specified name.

    ### Ancestors (in MRO)

    * unittest.case.TestCase

    ### Methods

    `test_load_benchmark_dataset_dataloader(self)`
    :

    `test_load_benchmark_dataset_raw_samples(self)`
    :

    `test_load_multiple_benchmark_splits(self)`
    :

    `test_registry_populated(self)`
    :

    `test_synthetic_sample_generation(self)`
    :