import os
import re
import glob

runners = sorted(glob.glob("/Users/tahamajs/Documents/uni/Project/Codes/runners/run_all_4_*.py"))
attentions = sorted(glob.glob("/Users/tahamajs/Documents/uni/Project/Codes/framework/attentions/*.py"))
compressors = sorted(glob.glob("/Users/tahamajs/Documents/uni/Project/Codes/framework/compression_engine/*.py"))

print(f"Total Runners: {len(runners)}")
print(f"Total Attention Modules: {len(attentions)}")
print(f"Total Compressor Modules: {len(compressors)}")

# Check for syntax errors across all python files
import py_compile
errors = []
for p in runners + attentions + compressors:
    try:
        py_compile.compile(p, doraise=True)
    except Exception as e:
        errors.append((p, str(e)))

print(f"Syntax Validation: {len(errors)} errors found.")
for f, err in errors:
    print(f"ERR: {f} -> {err}")

