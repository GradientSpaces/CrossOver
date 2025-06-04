#!/bin/bash

# Directory containing the zip files; update this as needed.
S3D_DOWNLOAD_DIR="/Users/gauravpradeep/CrossOver_ScaleUp"

# Find and sort all Structured3D_*.zip files in the directory
cd "$S3D_DOWNLOAD_DIR" || exit 1

for zip_file in Structured3D_*.zip; do
    # Skip if no files match
    [ -e "$zip_file" ] || continue
    extract_dir="${zip_file%.zip}"
    echo "Extracting $zip_file..."
    mkdir -p "$extract_dir"
    unzip -q "$zip_file" -d "$extract_dir"
done

echo "Done extracting all zips."