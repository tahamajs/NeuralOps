"""`test` command."""
import subprocess
import sys


def cmd_test():
    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests/test_all_attention_modules.py",
         "tests/test_all_compressor_modules.py", "-q"]
    )

