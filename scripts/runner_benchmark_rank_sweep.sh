#!/bin/bash
# Wrapper - redirects to unified CLI: python -m neuralops bench
MODULE=${1:-"asvd_attention"}
exec python3 -m neuralops bench --module "$MODULE" --mode eval "$@"
