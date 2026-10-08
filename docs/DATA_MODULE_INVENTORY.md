# Data Module Inventory

| Path | Lines | Classes | Functions | Sample Classes | Sample Functions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| neuralops/data/__init__.py | 87 | 0 | 0 |  |  |
| neuralops/data/augmentation.py | 27 | 0 | 2 |  | random_mask_augment, mask_tokens |
| neuralops/data/base.py | 367 | 3 | 2 | TaskType, DataConfig, ProductionDataCollator | preprocess_function, create_dataloader |
| neuralops/data/benchmark_datasets.py | 43 | 0 | 0 |  |  |
| neuralops/data/benchmark_datasets/__init__.py | 50 | 0 | 0 |  |  |
| neuralops/data/benchmark_datasets/loaders.py | 308 | 1 | 2 | BenchmarkTorchDataset | load_benchmark_dataset, get_synthetic_benchmark_sample |
| neuralops/data/benchmark_datasets/manifest.py | 328 | 2 | 5 | BenchmarkDatasetInfo, BenchmarkDatasetRegistry | register, get, is_registered |
| neuralops/data/benchmark_datasets/synthetic.py | 430 | 0 | 0 |  |  |
| neuralops/data/cached_tensors.py | 168 | 0 | 2 |  | get_cached_tokenized_dataset, tok_fn |
| neuralops/data/custom.py | 24 | 0 | 1 |  | load_custom_dataset |
| neuralops/data/custom_loader.py | 106 | 0 | 2 |  | load_custom_dataset, load_dataset_from_file |
| neuralops/data/dataset_discovery.py | 12 | 0 | 1 |  | list_local_datasets |
| neuralops/data/dataset_loader.py | 146 | 1 | 1 | ExperimentConfig | load_dataset |
| neuralops/data/glue.py | 67 | 0 | 2 |  | load_glue_dataset, load_superglue_dataset |
| neuralops/data/glue_loader.py | 212 | 0 | 6 |  | load_glue_split, load_glue_local, get_num_labels_and_is_regression |
| neuralops/data/multi_choice.py | 186 | 1 | 2 | MultiChoiceDataset | is_multichoice_dataset, convert_to_multichoice_format |
| neuralops/data/prepare_datasets.py | 496 | 1 | 8 | SyntheticDataset | prepare_glue_dataset, prepare_lm_dataset, create_benchmark_dataloader |
| neuralops/data/preprocessing.py | 33 | 0 | 1 |  | preprocess_function |
| neuralops/data/synthetic_generator.py | 397 | 2 | 5 | SyntheticPerplexityDataset, SyntheticClassificationDataset | generate_zipfian_tokens, generate_zipfian_markov_tokens, compute_empirical_entropy |
| neuralops/data/task_mapper.py | 57 | 0 | 3 |  | get_metric_name, get_num_labels, get_text_columns |
| neuralops/data/validation.py | 77 | 2 | 1 | ValidationReport, DataValidator | validate |
