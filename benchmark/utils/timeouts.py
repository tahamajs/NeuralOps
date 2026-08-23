"""Execution timeout decorators and context managers for preventing stalled benchmark tasks."""

from __future__ import annotations
import signal
import functools
from typing import Callable, Any


class TimeoutException(Exception):
    """Raised when an operation exceeds its allotted wall-clock runtime limit."""
    pass


def timeout(seconds: int) -> Callable:
    """
    Decorator enforcing a hard runtime execution limit via Unix signals.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            def _handle_timeout(signum, frame):
                raise TimeoutException(f"Execution timed out after {seconds} seconds.")

            # Signal only works on main thread in Unix-like platforms
            use_signal = hasattr(signal, "SIGALRM")
            if use_signal:
                old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                if use_signal:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator
