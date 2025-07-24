#!/usr/bin/env python3
"""Demo script to showcase AIDA TUI features."""

import subprocess
import sys
import time


def main():
    """Run TUI demo."""
    print("=== AIDA TUI Demo ===\n")
    print("This demo showcases the AIDA Text User Interface (TUI).")
    print("\nFeatures:")
    print("- Left panel (2/3): Chat interface")
    print("- Top right: Resource monitoring (CPU/Memory/GPU)")
    print("- Middle right: Ongoing tasks")
    print("- Bottom right: Available agents")
    print("\nKeyboard shortcuts:")
    print("- Ctrl+C: Quit")
    print("- Ctrl+L: Clear chat")
    print("- Ctrl+T: Toggle theme")
    print("\nStarting TUI in 3 seconds...")

    time.sleep(3)

    # Run the TUI
    try:
        subprocess.run([sys.executable, "-m", "aida.cli.main", "tui"])
    except KeyboardInterrupt:
        print("\nTUI closed.")


if __name__ == "__main__":
    main()
