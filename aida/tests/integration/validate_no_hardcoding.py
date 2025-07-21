"""Validate that no hardcoding exists for test questions."""

from pathlib import Path


def check_file_for_hardcoded_questions(filepath: Path) -> list:
    """Check a Python file for hardcoded test question handling."""
    violations = []

    with open(filepath) as f:
        content = f.read()

    # Test questions from our test suite
    test_questions = [
        "What is the capital of France?",
        "List all Python files in the current directory",
        "Create a Python script that prints 'Hello, AIDA!' and then run it",
        "test_output.txt",  # Common test file name
    ]

    # Check for direct string comparisons with test questions
    for question in test_questions:
        # Check for exact comparisons
        if f'== "{question}"' in content or f"== '{question}'" in content:
            violations.append(f"Direct comparison with test question: {question}")

        # Check for if conditions
        if f'if "{question}"' in content or f"if '{question}'" in content:
            violations.append(f"Conditional check for test question: {question}")

        # Check for special case handling
        if f'"{question}":' in content and ("return" in content or "response" in content):
            # Could be a hardcoded response mapping
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if (
                    f'"{question}"' in line
                    and ":" in line
                    and i + 1 < len(lines)
                    and ("return" in lines[i + 1] or "response" in lines[i + 1])
                ):
                    violations.append(f"Possible hardcoded response for: {question}")

    # Check for test mode conditionals
    test_patterns = [
        "if.*test_mode",
        "if.*test_run",
        "if.*is_test",
        "if.*context.get.*test",
    ]

    for pattern in test_patterns:
        if pattern.replace(".*", "") in content.lower():
            violations.append(f"Test mode conditional detected: {pattern}")

    return violations


def validate_no_hardcoding():
    """Validate the entire codebase for hardcoded test handling."""
    print("🔍 Validating No Hardcoding Policy")
    print("=" * 50)

    # Directories to check
    check_dirs = [
        "aida/core",
        "aida/tools",
        "aida/cli/commands",
        "aida/llm",
    ]

    total_violations = []
    files_checked = 0

    for dir_path in check_dirs:
        dir_full = Path(dir_path)
        if not dir_full.exists():
            continue

        print(f"\nChecking {dir_path}/...")

        for py_file in dir_full.rglob("*.py"):
            # Skip test files themselves
            if "test" in py_file.name or "__pycache__" in str(py_file):
                continue

            files_checked += 1
            violations = check_file_for_hardcoded_questions(py_file)

            if violations:
                total_violations.extend([(str(py_file), v) for v in violations])
                print(f"  ❌ {py_file.name}: {len(violations)} violations")
                for v in violations:
                    print(f"     - {v}")

    print("\n" + "=" * 50)
    print(f"Checked {files_checked} files")

    if total_violations:
        print(f"❌ FAILED: Found {len(total_violations)} hardcoding violations!")
        print("\nViolations by file:")
        for filepath, violation in total_violations:
            print(f"  {filepath}: {violation}")
        return False
    else:
        print("✅ SUCCESS: No hardcoding detected!")
        print("All test questions will be handled naturally by the system.")
        return True


if __name__ == "__main__":
    import sys

    success = validate_no_hardcoding()
    sys.exit(0 if success else 1)
