"""Local conftest for table serialization tests.

This conftest is minimal and doesn't depend on the main app,
allowing table serialization tests to run in isolation.
"""
import sys
from pathlib import Path

# Ensure app is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "app"))
