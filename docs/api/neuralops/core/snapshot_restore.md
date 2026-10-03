Module neuralops.core.snapshot_restore
======================================
Model State Snapshot & Exact Bit-Identical Restorer.

Preserves the exact structural integrity, module graph, parameter buffers,
and random state so attention modules can be dynamically swapped in-memory
with 100% exact bit-identical restoration and zero weight/state drift.

Classes
-------

`ModelStateSnapshot(model: nn.Module)`
:   Exact bit-identical snapshot of an nn.Module.

    ### Methods

    `restore(self) ‑> None`
    :   Restores the model structure and tensor values to exact bit-identical snapshot state.

    `verify_bit_identical(self) ‑> tuple[bool, float]`
    :   Verify that current model state is bit-identical to snapshot. Returns (is_identical, max_diff).