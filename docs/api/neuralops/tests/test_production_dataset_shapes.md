Module neuralops.tests.test_production_dataset_shapes
=====================================================
Production Dataset Shapes & Collation Verification Test Suite.

Verifies:
1. ProductionDataCollator batch padding to multiples of 8 (Tensor Core aligned).
2. Exact 2D tensor shapes [B, L] for input_ids and attention_mask.
3. Proper 1D/2D label shapes and dtype casting for Classification, Regression, and Causal LM.
4. MultiChoiceDataset 3D tensor shapes [B, num_choices, L].
5. Device and pinned memory tensor transfer compatibility.

Classes
-------

`TestProductionDatasetShapes()`
:   Test suite ensuring exact tensor dimensions, dtypes, and memory safety for production GPU servers.

    ### Methods

    `test_device_transfer_and_contiguous_memory(self) ‑> None`
    :   Verify all collated batch tensors are contiguous and support non-blocking device transfers.

    `test_multichoice_dataset_3d_shapes(self) ‑> None`
    :   Verify MultiChoiceDataset yields 3D tensors [B, num_choices, max_length].

    `test_production_collator_2d_language_modeling_labels(self) ‑> None`
    :   Verify 2D language modeling labels are padded with -100 ignore index.

    `test_production_data_collator_padding_and_alignment(self) ‑> None`
    :   Verify ProductionDataCollator pads sequences to multiples of 8 for GPU Tensor Cores.

    `test_regression_labels_float_dtype(self) ‑> None`
    :   Verify regression task labels produce float32 tensors.

    `test_synthetic_classification_dataset_shapes(self) ‑> None`
    :   Verify SyntheticClassificationDataset yields valid shapes [B, L] and [B].

    `test_synthetic_perplexity_dataset_shapes(self) ‑> None`
    :   Verify SyntheticPerplexityDataset yields valid shapes [B, L] for input_ids and labels.