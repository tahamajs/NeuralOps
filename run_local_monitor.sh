#!/usr/bin/env bash
# ==============================================================================
# Local Live Dashboard Monitor for NeuralOps Framework
# Serves the real-time analytics web studio locally at http://localhost:8080
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_DIR}/Codes:${PYTHONPATH}"

echo "======================================================================"
echo "⚡ Starting Local NeuralOps Live View Dashboard"
echo "   URL: http://localhost:8080"
echo "======================================================================"

python3 -u Codes/framework/live_monitor.py
