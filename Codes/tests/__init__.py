"""
Unit and Integration Test Suite for Efficient Transformer Toolkit & SVD Linear Attention Framework.
"""

import sys
import os

# Ensure framework package root is on sys.path
_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)
