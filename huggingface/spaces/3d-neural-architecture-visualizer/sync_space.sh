#!/usr/bin/env bash
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
#
# Sync and deploy 3D Neural Architecture Visualizer to Hugging Face Spaces:
# https://huggingface.co/spaces/Hooshaai/3d-neural-architecture-visualizer

set -euo pipefail

REPO_ID="${1:-Hooshaai/3d-neural-architecture-visualizer}"
TARGET_DIR="${2:-/tmp/hf-3d-visualizer-space}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "🌌 Hugging Face Space Git Sync: ${REPO_ID}"
echo "========================================================================"

# Prepare clean target directory
rm -rf "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

echo "📥 Cloning Space repository..."
if [ -n "${HF_TOKEN:-}" ]; then
    git clone "https://user:${HF_TOKEN}@huggingface.co/spaces/${REPO_ID}" "${TARGET_DIR}" || {
        echo "⚠️ Repository not found or empty. Initializing new local git repo..."
        mkdir -p "${TARGET_DIR}"
        cd "${TARGET_DIR}"
        git init
        git remote add origin "https://user:${HF_TOKEN}@huggingface.co/spaces/${REPO_ID}"
    }
else
    git clone "https://huggingface.co/spaces/${REPO_ID}" "${TARGET_DIR}" || {
        echo "⚠️ Repository not found or empty. Initializing new local git repo..."
        mkdir -p "${TARGET_DIR}"
        cd "${TARGET_DIR}"
        git init
        git remote add origin "https://huggingface.co/spaces/${REPO_ID}"
    }
fi

echo "📦 Synchronizing static web distribution files..."
cd "${SCRIPT_DIR}"

# Copy all static web bundle assets
cp README.md index.html app.js style.css spatial-data.json \
   modules-catalog.json modules_components.json standalone.html \
   latent_manifold.html favicon.ico "${TARGET_DIR}/"

# Copy directories
cp -R src components "${TARGET_DIR}/"

cd "${TARGET_DIR}"

# Configure git commit attribution strictly conforming to AGENTS.md
git config user.name "taha"
git config user.email "tahamaj4@gmail.com"

# Check if there are changes
git add -A
if git diff-index --quiet HEAD 2>/dev/null; then
    echo "ℹ️ No changes detected. Space is already up to date!"
else
    echo "💾 Creating commit..."
    git commit --author="taha <tahamaj4@gmail.com>" -m "feat: synchronize 3D Neural Architecture Visualizer (241 modules, Three.js WebGL)"
    echo "🚀 Pushing to https://huggingface.co/spaces/${REPO_ID}..."
    git branch -M main
    git push -u origin main
    echo "✨ Successfully deployed to https://huggingface.co/spaces/${REPO_ID}"
fi
