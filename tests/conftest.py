"""Shared fixtures for the repository-level test suite.

Tests run from the repository root with ``python3 -m pytest tests/ -q``.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
