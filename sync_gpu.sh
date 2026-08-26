#!/usr/bin/env bash
# ==============================================================================
# GPU Code Synchronization Script
# Efficiently syncs local Codes/ directory to remote GPU server at D:\SVD_Project\
# Uses streaming tar over SSH with full exclusion of caches, build files, and logs.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "================================================================================"
echo "🚀 SYNCING CODE TO GPU SERVER (gpu -> D:\\SVD_Project\\Codes\\)"
echo "================================================================================"

EXCLUDES=(
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='*.pyo'
    --exclude='.git'
    --exclude='.pytest_cache'
    --exclude='.mypy_cache'
    --exclude='.venv'
    --exclude='venv'
    --exclude='node_modules'
    --exclude='results'
    --exclude='experiments'
    --exclude='logs'
    --exclude='*.log'
    --exclude='*.tmp'
    --exclude='*.pdf'
    --exclude='*.png'
)

echo "📦 Packaging and streaming Codes/ directory..."
tar -cf - "${EXCLUDES[@]}" Codes | ssh gpu "tar -xf - -C D:/SVD_Project"

echo "✅ Sync complete! Remote D:\\SVD_Project\\Codes\\ is up to date."
echo "================================================================================"
