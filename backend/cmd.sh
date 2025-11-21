#!/bin/bash

# cmd.sh command script for shortcut commands 
#
# Usage:
#   ./cmd.sh -e or --export     - export dependencies from pyproject.toml

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--export)
      echo "Exporting dependencies from pyproject.toml to requirements.txt..."
      uv pip compile pyproject.toml -o requirements.txt && exit 0
      ;;
    *)
      echo "NO command available '$1'"
      exit 1
      ;;
  esac

done