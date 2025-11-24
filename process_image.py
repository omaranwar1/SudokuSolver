#!/usr/bin/env python3
"""
Convenience script to process Sudoku images.

This script provides a simple interface to the Sudoku Solver pipeline.

Usage:
    python process_image.py 01.jpg
    python process_image.py path/to/sudoku.jpg --output my_output/
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sudoku_solver import main

if __name__ == '__main__':
    main()
