"""
Simple backtracking Sudoku solver with basic safety checks.
"""

import numpy as np


def _find_empty(board: np.ndarray):
    positions = np.argwhere(board == 0)
    if positions.size == 0:
        return None
    return tuple(positions[0])


def _is_valid(board: np.ndarray, row: int, col: int, val: int) -> bool:
    if val in board[row, :]:
        return False
    if val in board[:, col]:
        return False

    r0 = (row // 3) * 3
    c0 = (col // 3) * 3
    if val in board[r0:r0 + 3, c0:c0 + 3]:
        return False

    return True


def _validate_givens(board: np.ndarray) -> tuple[bool, str]:
    """Check for duplicate givens; fails fast to avoid long searches."""
    for i in range(9):
        row_vals = [v for v in board[i, :] if v != 0]
        if len(row_vals) != len(set(row_vals)):
            return False, f"Row {i+1} has duplicate given digit"

        col_vals = [v for v in board[:, i] if v != 0]
        if len(col_vals) != len(set(col_vals)):
            return False, f"Column {i+1} has duplicate given digit"

    for br in range(3):
        for bc in range(3):
            block = board[br*3:(br+1)*3, bc*3:(bc+1)*3].ravel()
            block_vals = [v for v in block if v != 0]
            if len(block_vals) != len(set(block_vals)):
                return False, f"3x3 block ({br+1},{bc+1}) has duplicate given digit"

    return True, ""


def solve_board(board: np.ndarray, step_counter: list[int], max_steps: int) -> bool:
    """In-place backtracking solver. Returns True if solved."""
    if step_counter[0] > max_steps:
        return False

    empty = _find_empty(board)
    if empty is None:
        return True

    r, c = empty
    for val in range(1, 10):
        if _is_valid(board, r, c, val):
            board[r, c] = val
            step_counter[0] += 1
            if solve_board(board, step_counter, max_steps):
                return True
            board[r, c] = 0

    return False


def solve_puzzle(board: np.ndarray, max_steps: int = 200000) -> tuple[np.ndarray | None, str]:
    """
    Return a solved copy of the board, or (None, reason) if unsolvable or invalid.
    Limits the search to max_steps expansions to avoid runaway loops.
    """
    is_valid, reason = _validate_givens(board)
    if not is_valid:
        return None, reason

    working = board.copy()
    steps = [0]
    solved = solve_board(working, steps, max_steps)
    if solved:
        return working, f"Solved in {steps[0]} steps"
    if steps[0] >= max_steps:
        return None, f"Stopped after {steps[0]} steps (limit {max_steps})"
    return None, "No solution found"
