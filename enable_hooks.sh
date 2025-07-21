#!/bin/bash
# Enable pre-commit hooks for the AIDA project

echo "=== Enabling Git Hooks for AIDA ==="
echo

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install

# Also install commit-msg hook for commitizen
echo "Installing commit-msg hook..."
uv run pre-commit install --hook-type commit-msg

echo
echo "✅ Git hooks have been enabled!"
echo
echo "The following checks will run automatically:"
echo "  - Before commit: Ruff formatting, Ty type checking, security scans"
echo "  - Commit message: Commitizen format validation"
echo
echo "To run hooks manually: uv run pre-commit run --all-files"
echo "To skip hooks (emergency only): git commit --no-verify"
