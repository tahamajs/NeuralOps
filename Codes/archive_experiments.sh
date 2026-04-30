#!/usr/bin/env bash
# archive_experiments.sh
# Compress all experiment subdirectories under ./experiments into individual tar.gz files.

set -euo pipefail  # strict mode

SOURCE_DIR="./experiments"
OUTPUT_DIR="./archived_experiments"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find all experiment directories (directly under experiments/ with a name like 2026*)
# We use -maxdepth 1 to avoid recursing into subdirectories of each experiment.
for exp_dir in "$SOURCE_DIR"/2026*/ ; do
    # Remove trailing slash
    exp_dir=${exp_dir%/}
    # Get the base name (e.g., 20260501-004923-1e8288ad)
    exp_name=$(basename "$exp_dir")
    
    # Define output archive name
    archive_name="$OUTPUT_DIR/experiment_${exp_name}.tar.gz"
    
    echo "Compressing $exp_dir -> $archive_name"
    # Use tar with gzip compression, preserving permissions and structure
    tar -czf "$archive_name" -C "$SOURCE_DIR" "$exp_name"
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: $archive_name"
    else
        echo "❌ Failed: $exp_dir"
    fi
done

echo "All archives created in $OUTPUT_DIR"
echo "Total size:"
du -sh "$OUTPUT_DIR"