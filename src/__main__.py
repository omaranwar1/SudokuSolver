"""
Entry point for running the sudoku_solver module as a package.

Usage:
    python -m src.sudoku_solver --image path/to/image.jpg
"""

from .sudoku_solver import main

if __name__ == '__main__':
    main()
