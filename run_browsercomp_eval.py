#!/usr/bin/env python3
"""
Simple script to run BrowserComp evaluation.

Usage:
    python run_browsercomp_eval.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from eval.run_browsercomp import main

if __name__ == "__main__":
    print("🚀 Running BrowserComp Evaluation")
    print("=" * 40)
    main() 