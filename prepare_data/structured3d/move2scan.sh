#!/bin/bash

# Define the base directory (current directory in this case)
BASE_DIR="/Users/gauravpradeep/CrossOver_ScaleUp/Structured3D"   

# Define the target subfolder
SCANS_DIR="$BASE_DIR/scans"

# Create the scans folder if it doesn't exist
mkdir -p "$SCANS_DIR"

# Move all files and directories (except "scans" itself) into the scans folder
for item in "$BASE_DIR"/*; do
  # Skip the scans directory
  if [[ "$(basename "$item")" == "scans" ]]; then
    continue
  fi

  # Move the item into the scans folder
  echo "Moving $item to $SCANS_DIR"
  mv "$item" "$SCANS_DIR"
done

echo "All files and directories have been moved to the 'scans' folder."