#!/bin/bash

# Loop through all files that start with "slurm" in the current directory
for file in slurm*; do
    # Check if the file exists (in case no files match the pattern)
    if [[ -f "$file" ]]; then
        echo "Displaying last 10 lines of $file:"
        # Use tail to print the last 10 lines of the file
        tail -n 10 "$file"
        echo "----------------------------------"
    fi
done