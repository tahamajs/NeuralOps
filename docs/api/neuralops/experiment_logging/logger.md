Module neuralops.experiment_logging.logger
==========================================
Structured and thread-safe logger setup with JSON and human-readable formats.
Implements QueueHandler / QueueListener for non-blocking logging across threads and
ensures each DDP worker rank writes to an isolated log file (rank_{LOCAL_RANK}.log).

Functions
---------

`get_logger(name: str = 'neuralops') ‑> logging.Logger`
:   Return a logger for the given name.

`get_rank(configured_rank: int | None = None) ‑> int`
:   Determine the current process rank in single-node or DDP multi-GPU environments.
    
    Args:
        configured_rank: Explicit rank if specified in config.
    
    Returns:
        int: Process local/global rank (0 for single-node / master).

`setup_logging(config: LoggingConfig | None = None) ‑> logging.Logger`
:   Create and configure a thread-safe logger with isolated rank logging.
    
    Args:
        config: LoggingConfig dataclass specifying log parameters.
    
    Returns:
        logging.Logger: Configured logger instance.

`shutdown_logging() ‑> None`
:   Stop all background QueueListener workers and flush pending log events.

Classes
-------

`StructuredFormatter(rank: int = 0)`
:   JSON-line formatter for structured logging with rank metadata.
    
    Initialize the formatter with specified format strings.
    
    Initialize the formatter either with the specified format string, or a
    default as described above. Allow for specialized date formatting with
    the optional datefmt argument. If datefmt is omitted, you get an
    ISO8601-like (or RFC 3339-like) format.
    
    Use a style parameter of '%', '{' or '$' to specify that you want to
    use one of %-formatting, :meth:`str.format` (``{}``) formatting or
    :class:`string.Template` formatting in your format string.
    
    .. versionchanged:: 3.2
       Added the ``style`` parameter.

    ### Ancestors (in MRO)

    * logging.Formatter

    ### Methods

    `format(self, record: logging.LogRecord) ‑> str`
    :   Format log record into JSON string.