#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``unml-interface`` command."""

try:
    from _bootstrap import configure_runtime
except ModuleNotFoundError:
    from scripts._bootstrap import configure_runtime

configure_runtime()

from interface.cli import main


if __name__ == "__main__":
    main()
