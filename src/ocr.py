"""
OCR helpers using simple pattern matching for Sudoku digits.

Approach:
1) Split the straightened binary grid into 9x9 cells.
2) Discard empty cells using pixel coverage.
3) For remaining cells, resize the digit mask and compare against
   generated template digits using normalized cross correlation.
"""

import cv2
import numpy as np


def build_digit_templates(size: int = 28, thicknesses: tuple[int, ...] = (1, 2, 3)) -> dict[int, list[np.ndarray]]:
    """Generate clean digit templates (1-9) rendered with multiple stroke widths."""
    templates: dict[int, list[np.ndarray]] = {}
    font = cv2.FONT_HERSHEY_SIMPLEX

    for digit in range(1, 10):
        variants = []
        for thickness in thicknesses:
            canvas = np.zeros((size, size), dtype=np.uint8)
            cv2.putText(
                canvas,
                str(digit),
                (size // 6, int(size * 0.8)),
                font,
                1.2,
                255,
                thickness,
                cv2.LINE_AA,
            )
            variants.append(canvas)
        templates[digit] = variants

    return templates


def remove_grid_lines(binary_grid: np.ndarray) -> np.ndarray:
    """
    Remove most grid lines to reduce interference for OCR.

    Works on binary images where digits/lines are white on black.
    """
    h, w = binary_grid.shape
    cell_h, cell_w = h // 9, w // 9

    work = binary_grid.copy()
    if work.mean() > 127:
        work = cv2.bitwise_not(work)

    # Detect horizontal and vertical lines using morphology
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, cell_w // 2), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, cell_h // 2)))

    horiz = cv2.morphologyEx(work, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vert = cv2.morphologyEx(work, cv2.MORPH_OPEN, v_kernel, iterations=1)

    lines = cv2.bitwise_or(horiz, vert)

    # Dilate a little to cover line thickness
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)

    digits_only = cv2.bitwise_and(work, cv2.bitwise_not(lines))
    return digits_only


def _extract_digit_from_cell(cell: np.ndarray, templates: dict[int, list[np.ndarray]],
                             min_fill: float = 0.004,
                             accept_threshold: float = 0.30) -> tuple[int, float]:
    """
    Recognize a digit from a single cell image.

    Returns:
        (digit, score) where digit=0 means empty cell.
    """
    h, w = cell.shape
    margin = max(1, int(min(h, w) * 0.05))
    inner = cell[margin:h - margin, margin:w - margin]

    # Normalize polarity to make digits white on black
    work = inner.copy()
    if work.mean() > 127:
        work = cv2.bitwise_not(work)

    work = cv2.GaussianBlur(work, (3, 3), 0)
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Suppress residual grid bleed by clearing a small border
    border = max(2, int(min(h, w) * 0.08))
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

    # Connected components to isolate the digit blob
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0, 0.0

    # Skip background (label 0); choose largest remaining component
    digit_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[digit_label, cv2.CC_STAT_AREA]
    min_area = min_fill * (binary.shape[0] * binary.shape[1])
    if area < min_area or area > 0.7 * binary.size:
        return 0, 0.0

    x, y, bw, bh = stats[digit_label, cv2.CC_STAT_LEFT], stats[digit_label, cv2.CC_STAT_TOP], \
        stats[digit_label, cv2.CC_STAT_WIDTH], stats[digit_label, cv2.CC_STAT_HEIGHT]

    pad = 2
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(binary.shape[1], x + bw + pad), min(binary.shape[0], y + bh + pad)
    digit_roi = binary[y1:y2, x1:x2]

    # Square-pad before resize to maintain aspect
    side = max(digit_roi.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - digit_roi.shape[0]) // 2
    x_off = (side - digit_roi.shape[1]) // 2
    square[y_off:y_off + digit_roi.shape[0], x_off:x_off + digit_roi.shape[1]] = digit_roi

    target_size = next(iter(templates.values()))[0].shape[0]
    digit_norm = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Try both normal and inverted ROI to tolerate polarity issues
    candidates = [digit_norm, cv2.bitwise_not(digit_norm)]

    best_digit, best_score = 0, -1.0
    for candidate in candidates:
        for digit, tmpl_list in templates.items():
            for tmpl in tmpl_list:
                score = cv2.matchTemplate(candidate, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
                if score > best_score:
                    best_digit, best_score = digit, float(score)

    if best_score < accept_threshold:
        return 0, best_score

    return best_digit, best_score


def resolve_conflicts(board: np.ndarray, scores: np.ndarray, min_score_keep: float = 0.0
                      ) -> tuple[np.ndarray, list[str]]:
    """
    Remove duplicate givens in rows/cols/blocks by keeping only the highest-score hit.

    Returns:
        cleaned_board, list of notes about removed cells
    """
    cleaned = board.copy()
    notes: list[str] = []

    def drop_duplicates(index_list: list[tuple[int, int]], label: str):
        nonlocal cleaned, notes
        values = {}
        for r, c in index_list:
            v = cleaned[r, c]
            if v == 0:
                continue
            values.setdefault(v, []).append((r, c))
        for v, cells in values.items():
            if len(cells) <= 1:
                continue
            keep = max(cells, key=lambda p: scores[p])
            for cell in cells:
                if cell == keep:
                    continue
                if scores[cell] >= min_score_keep:
                    notes.append(f"{label}: dropped duplicate '{v}' at ({cell[0]+1},{cell[1]+1})")
                cleaned[cell] = 0

    # Rows
    for r in range(9):
        idx = [(r, c) for c in range(9)]
        drop_duplicates(idx, f"Row {r+1}")

    # Columns
    for c in range(9):
        idx = [(r, c) for r in range(9)]
        drop_duplicates(idx, f"Col {c+1}")

    # Blocks
    for br in range(3):
        for bc in range(3):
            idx = []
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    idx.append((r, c))
            drop_duplicates(idx, f"Block {br+1},{bc+1}")

    return cleaned, notes


def extract_grid_digits(binary_grid: np.ndarray, templates: dict[int, list[np.ndarray]] | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a 9x9 integer grid (0 = empty) from the straightened binary image.

    Returns:
        board (9x9 ints), scores (9x9 floats correlation confidence)
    """
    if templates is None:
        templates = build_digit_templates()

    # Prepare two views: with grid lines removed, and the raw inverted
    digits_only = remove_grid_lines(binary_grid)
    raw_view = binary_grid.copy()
    if raw_view.mean() > 127:
        raw_view = cv2.bitwise_not(raw_view)

    h, w = digits_only.shape
    cell_h = h // 9
    cell_w = w // 9

    board = np.zeros((9, 9), dtype=int)
    scores = np.zeros((9, 9), dtype=float)

    for r in range(9):
        for c in range(9):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell1 = digits_only[y1:y2, x1:x2]
            cell2 = raw_view[y1:y2, x1:x2]
            d1, s1 = _extract_digit_from_cell(cell1, templates)
            d2, s2 = _extract_digit_from_cell(cell2, templates)
            if s2 > s1:
                digit, score = d2, s2
            else:
                digit, score = d1, s1
            board[r, c] = digit
            scores[r, c] = score

    return board, scores


def format_board(board: np.ndarray) -> str:
    """Render the 9x9 board as a human-friendly string."""
    lines = []
    for r, row in enumerate(board):
        parts = []
        for c, val in enumerate(row):
            parts.append(str(val) if val != 0 else ".")
            if c in {2, 5}:
                parts.append("|")
        line = " ".join(parts)
        lines.append(line)
        if r in {2, 5}:
            lines.append("-" * len(line))
    return "\n".join(lines)


def render_solution_on_image(image: np.ndarray, solved: np.ndarray, original: np.ndarray) -> np.ndarray:
    """
    Overlay solved digits on top of the straightened image.

    Given digits (original) are drawn in white, solved digits in green.
    """
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    h, w = canvas.shape[:2]
    cell_h = h // 9
    cell_w = w // 9
    font = cv2.FONT_HERSHEY_SIMPLEX

    for r in range(9):
        for c in range(9):
            val = int(solved[r, c])
            if val == 0:
                continue
            color = (255, 255, 255) if original[r, c] != 0 else (0, 200, 0)
            text = str(val)
            size, _ = cv2.getTextSize(text, font, 0.9, 2)
            x = c * cell_w + (cell_w - size[0]) // 2
            y = r * cell_h + (cell_h + size[1]) // 2
            cv2.putText(canvas, text, (x, y), font, 0.9, color, 2, cv2.LINE_AA)

    return canvas
