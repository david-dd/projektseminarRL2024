#!/bin/bash

# Define the target directory
target_dir="$ZIH_USER_DIR/projects/projektseminarRL2024/experiments"

# Expand the variable to get the full path
expanded_dir=$(eval echo "$target_dir")

# Check if the directory exists
if [[ -d "$expanded_dir" ]]; then
    echo "$expanded_dir"
    echo
    # Loop through all directories in the target directory
    for folder in "$expanded_dir"/*/; do
        # Check if the entry is a directory
        if [[ -d "$folder" ]]; then
            # Print the folder name
            echo "EXPERIMENT_NAME=\"$(basename "$folder")\""
            echo
            # Loop through subfolders one level deeper
            for subfolder in "$folder"*/; do
                # Check if the entry is a directory
                if [[ -d "$subfolder" ]]; then
                    # Print the subfolder name with indentation
                    echo "    EXPERIMENT_NAME2=\"$(basename "$folder")\""
                    echo "    EXPERIMENT_SUBFOLDER=\"$(basename "$subfolder")\""
                    echo
                fi
            done
        fi
    done
else
    echo "Error: Directory $expanded_dir does not exist."
fi
