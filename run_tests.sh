#!/bin/bash

set -e  # Exit immediately if any command fails

# Install the package in editable mode
pip install -e .

# Delete test_results.txt if it exists
if [ -f test_results.txt ]; then
    rm test_results.txt
fi

# Run all tests and append output to a single file
echo "Running regular tests..."
pytest tests >> test_results.txt

echo "Running tests with coverage..."
pytest --cov=src tests >> test_results.txt

echo "Running tests with flake8..."
pytest --cache-clear --flake8 >> test_results.txt

echo "Tests complete. Results are in test_results.txt."
