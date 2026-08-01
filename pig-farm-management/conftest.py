"""Ensure the package is importable when running pytest from any directory."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
