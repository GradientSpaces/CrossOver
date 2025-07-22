#!/bin/bash

DATA_DIR="/Users/gauravpradeep/CrossOver_ScaleUp/extracted/Structured3D_bbox/Structured3D"  #this should be where all the data was moved-basically the structured3d dir within s3d_bbox 
TARGET_DIR="/Users/gauravpradeep/CrossOver_ScaleUp/Structured3D"  #this should be the final structured3d dir where you want to move the scans
# Define the target subfolder
SCANS_DIR="$TARGET_DIR/scans"

# Create the scans folder if it doesn't exist
mkdir -p "$SCANS_DIR"

# Move all files and directories (except "scans" itself) into the scans folder
for item in "$DATA_DIR"/*; do
  # Skip the scans directory
  if [[ "$(basename "$item")" == "scans" ]]; then
    continue
  fi

  # Move the item into the scans folder
  echo "Moving $item to $SCANS_DIR"
  mv "$item" "$SCANS_DIR"
done

echo "All files and directories have been moved to the 'scans' folder."