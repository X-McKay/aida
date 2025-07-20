#!/bin/bash
# Test CI steps locally

echo "=== Testing CI/CD Steps Locally ==="
echo "This simulates the key steps from our GitHub Actions workflow"
echo

# Set environment variables
export PYTHON_VERSION="3.11"
export UV_VERSION="0.7.13"

echo "1. Setting up UV..."
if ! command -v uv &> /dev/null; then
    echo "UV not found. Please install UV first."
    exit 1
fi

echo "2. Installing Python $PYTHON_VERSION..."
uv python install $PYTHON_VERSION

echo "3. Installing dependencies..."
uv sync --all-extras

echo "4. Running smoke tests..."
uv run python tests/scripts/smoke_test.py

echo "5. Running integration tests..."
for suite in hybrid_files hybrid_system hybrid_execution hybrid_context; do
    echo "Testing $suite..."
    uv run python -m aida.cli.main test run --suite $suite || exit 1
done

echo
echo "=== All CI steps completed successfully! ==="
