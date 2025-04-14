#!/bin/bash

# --- Script to unzip all zip files in a directory into subdirectories ---

# Check if an input directory path was provided
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/your/zip/files"
  exit 1
fi

# Assign the first argument to a variable
TARGET_DIR="$1"

# Check if the provided path is actually a directory
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: '$TARGET_DIR' is not a valid directory."
  exit 1
fi

# Check if the unzip command is available
if ! command -v unzip &> /dev/null; then
    echo "Error: 'unzip' command not found. Please install it (e.g., sudo apt update && sudo apt install unzip)."
    exit 1
fi


echo "Processing directory: $TARGET_DIR"

# Change to the target directory (or operate using full paths)
# Using pushd/popd is safer than just cd
pushd "$TARGET_DIR" > /dev/null || { echo "Error: Could not change to directory '$TARGET_DIR'"; exit 1; }

# Loop through all files ending with .zip in the current directory
for zipfile in *.zip; do
  # Check if the zipfile variable actually points to a file
  # This handles the case where no *.zip files are found
  [ -f "$zipfile" ] || continue

  # Get the base name of the file without the .zip extension
  dirname=$(basename "$zipfile" .zip)

  echo "Processing '$zipfile' -> '$dirname/'"

  # Create the directory; -p ensures no error if it already exists
  # and creates parent directories if needed (though not applicable here)
  mkdir -p "$dirname"

  # Unzip the file into the created directory
  # -q makes unzip quiet (less output)
  # -d specifies the destination directory
  unzip -q "$zipfile" -d "$dirname"

  # Optional: Check if unzip was successful
  if [ $? -ne 0 ]; then
    echo "Warning: Failed to unzip '$zipfile' properly."
  fi
done

# Return to the original directory
popd > /dev/null

echo "Finished processing all zip files."
exit 0