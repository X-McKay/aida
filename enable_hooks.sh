#!/usr/bin/env bash
# Enable git hooks for this repository

echo "Enabling AIDA git hooks..."
git config core.hooksPath .githooks
echo "✅ Git hooks enabled"
echo ""
echo "The pre-commit hook will now run smoke tests before every commit."
echo "To disable: git config --unset core.hooksPath"