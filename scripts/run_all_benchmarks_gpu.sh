#!/bin/bash
# Wrapper - redirects to unified CLI: python -m neuralops bench
exec python3 -m neuralops bench "$@"
