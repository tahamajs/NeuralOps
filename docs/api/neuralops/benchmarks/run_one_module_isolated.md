Module neuralops.benchmarks.run_one_module_isolated
===================================================
Child runner: executes exactly one module evaluation.
Called by run_quality_sweep.py via subprocess.Popen.
Writes result JSON to --output_json path and exits with code 0 or 1.

Functions
---------

`main()`
: