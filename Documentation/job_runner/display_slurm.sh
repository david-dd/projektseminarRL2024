#!/bin/bash

# Get the current directory path using pwd
current_dir=$(pwd)

# Loop through all files that start with "slurm" in the current directory
for file in slurm*; do
    # Check if the file exists (in case no files match the pattern)
    if [[ -f "$file" ]]; then
         
        echo
        echo "-----------------------------------------------------"
        echo "$current_dir/$file"
        echo "-----------------------------------------------------"
        echo
        # Use tail to print the last 10 lines of the file
        tail -n 22 "$file"
        echo
        echo
    fi
done