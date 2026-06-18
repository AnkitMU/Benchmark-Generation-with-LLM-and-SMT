"""Pytest root conftest — adds the project root to sys.path so that
``import modules.*`` works in all test files without per-file path hacks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
