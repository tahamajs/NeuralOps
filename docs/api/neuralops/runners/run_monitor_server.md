Module neuralops.runners.run_monitor_server
===========================================
Production-grade NeuralOps Telemetry, Benchmark Database & Documentation Studio Server
Serves the neuralops/web application on port 8080 with 100% code-accurate metrics
and an interactive Documentation Reader API.

Functions
---------

`build_aggregated_stats()`
:   

`get_doc_content(mod_query)`
:   Retrieves the Markdown documentation for a requested module or topic.

`get_live_gpu_telemetry()`
:   

`get_live_logs(source='active', module=None, lines=200)`
:   Fetches real-time log lines from the GPU server via gpu_live_probe or local disk.

`get_loss_history(module=None, model=None)`
:   Returns step loss history array and current step for active or requested module.

`get_module_detail(mod_query)`
:   Returns comprehensive run-level details, loss trajectory, and dataset breakdown for a specific module.

`get_module_doc_index()`
:   Builds an index of available markdown documentation across Docs/content/.

`launch_new_run(params)`
:   Queues a new benchmark experiment with SVD & LoRA parameters.

`load_harvested_benchmark_records()`
:   Reads all historical + GPU empirical benchmark runs with exact code-accurate metrics.

`stop_current_run()`
:   Safely halts the current sweep process on GPU server.

Classes
-------

`ComprehensiveStudioHandler(*args, **kwargs)`
:   Simple HTTP request handler with GET and HEAD commands.
    
    This serves files from the current directory and any of its
    subdirectories.  The MIME type for files is determined by
    calling the .guess_type() method.
    
    The GET and HEAD requests are identical except that the HEAD
    request omits the actual contents of the file.

    ### Ancestors (in MRO)

    * http.server.SimpleHTTPRequestHandler
    * http.server.BaseHTTPRequestHandler
    * socketserver.StreamRequestHandler
    * socketserver.BaseRequestHandler

    ### Methods

    `do_GET(self)`
    :   Serve a GET request.

    `do_POST(self)`
    :