"""
OCR helpers using simple pattern matching for Sudoku digits.

Approach:
1) Split the straightened binary grid into 9x9 cells.
2) Discard empty cells using pixel coverage.
3) For remaining cells, resize the digit mask and compare against
   generated template digits using normalized cross correlation.
"""

import os
import cv2
import numpy as np


def build_digit_templates(
    size: int = 28,
    thicknesses: tuple[int, ...] = (1, 2),
    rotations: tuple[int, ...] = (-3, 0, 3),
    fonts: tuple[int, ...] = (
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
    ),
    scales: tuple[float, ...] = (1.0, 1.1),
) -> dict[int, list[np.ndarray]]:
    """
    Generate digit templates (1-9) with multiple fonts, stroke widths, rotations,
    and slight morphological tweaks.
    """
    templates: dict[int, list[np.ndarray]] = {}
    kernel = np.ones((2, 2), np.uint8)

    for digit in range(1, 10):
        variants = []
        for font in fonts:
            for thickness in thicknesses:
                for scale in scales:
                    canvas = np.zeros((size, size), dtype=np.uint8)
                    cv2.putText(
                        canvas,
                        str(digit),
                        (size // 6, int(size * 0.8)),
                        font,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA,
                    )
                    for angle in rotations:
                        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
                        rotated = cv2.warpAffine(canvas, M, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
                        variants.append(rotated)
                        variants.append(cv2.erode(rotated, kernel, iterations=1))
                        variants.append(cv2.dilate(rotated, kernel, iterations=1))
        templates[digit] = variants

    return templates


def _build_template_contours(templates: dict[int, list[np.ndarray]]) -> dict[int, list[np.ndarray | None]]:
    """Precompute largest contour for each template variant."""
    contours: dict[int, list[np.ndarray | None]] = {}
    for digit, imgs in templates.items():
        digit_contours: list[np.ndarray | None] = []
        for img in imgs:
            cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                digit_contours.append(None)
                continue
            digit_contours.append(max(cnts, key=cv2.contourArea))
        contours[digit] = digit_contours
    return contours


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


def _extract_digit_from_cell(
    cell: np.ndarray,
    templates: dict[int, list[np.ndarray]],
    min_fill: float = 0.003,
    accept_threshold: float = 0.20,
    min_separation: float = 0.06,
    fallback: bool = False,
    debug: bool = False,
) -> tuple[int, float]:
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

    def make_binary(img: np.ndarray) -> np.ndarray:
        _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=1)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, iterations=1)
        return b

    binary = make_binary(work)

    # Suppress residual grid bleed by clearing a small border
    border = max(2, int(min(h, w) * 0.08))
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

    # Quick empty check based on overall ink coverage
    coverage = binary.mean() / 255.0
    coverage_floor = 0.003 if fallback else 0.005
    if coverage < coverage_floor:
        # Try inverted polarity before giving up
        inv = make_binary(cv2.bitwise_not(work))
        coverage_inv = inv.mean() / 255.0
        if coverage_inv > coverage:
            binary = inv
            coverage = coverage_inv
            if debug:
                print(f"[OCR] used inverted binary, coverage={coverage:.4f}")
        else:
            if debug:
                print(f"[OCR] reject by coverage={coverage:.4f}")
            return 0, 0.0

    # Connected components to isolate the digit blob
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        # Adaptive fallback
        adapt = cv2.adaptiveThreshold(
            work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        adapt = cv2.bitwise_not(adapt)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(adapt, connectivity=8)
        if num_labels <= 1:
            if debug:
                print("[OCR] no blobs even after adaptive")
            return 0, 0.0
        binary = adapt

    # Skip background (label 0); choose largest remaining component
    digit_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    area = stats[digit_label, cv2.CC_STAT_AREA]
    min_area = min_fill * (binary.shape[0] * binary.shape[1])
    area_high = 0.90 if fallback else 0.80
    if area < min_area or area > area_high * binary.size:
        if debug:
            print(f"[OCR] reject by area: {area/binary.size:.3f}")
        return 0, 0.0

    x, y, bw, bh = (
        stats[digit_label, cv2.CC_STAT_LEFT],
        stats[digit_label, cv2.CC_STAT_TOP],
        stats[digit_label, cv2.CC_STAT_WIDTH],
        stats[digit_label, cv2.CC_STAT_HEIGHT],
    )

    # Filter out blobs that are too skinny/wide or too short (likely grid scraps)
    aspect = bw / max(1, bh)
    aspect_low = 0.15 if fallback else 0.2
    aspect_high = 7.0 if fallback else 6.0
    if aspect < aspect_low or aspect > aspect_high:
        if debug:
            print(f"[OCR] reject by aspect: {aspect:.2f}")
        return 0, 0.0
    min_h = 0.15 if fallback else 0.20
    min_w = 0.07 if fallback else 0.10
    if bh < min_h * binary.shape[0] or bw < min_w * binary.shape[1]:
        if debug:
            print(f"[OCR] reject by size: bw={bw}, bh={bh}")
        return 0, 0.0

    # Mask out everything except the chosen blob to ignore stray noise
    mask = np.zeros_like(binary)
    mask[labels == digit_label] = 255
    pad = 2
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(binary.shape[1], x + bw + pad), min(binary.shape[0], y + bh + pad)
    digit_roi = mask[y1:y2, x1:x2]

    # Square-pad before resize to maintain aspect
    side = max(digit_roi.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - digit_roi.shape[0]) // 2
    x_off = (side - digit_roi.shape[1]) // 2
    square[y_off:y_off + digit_roi.shape[0], x_off:x_off + digit_roi.shape[1]] = digit_roi

    target_size = next(iter(templates.values()))[0].shape[0]
    digit_norm = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Candidate representations: normal/inverted Otsu plus stroke tweaks
    candidates = [digit_norm, cv2.bitwise_not(digit_norm)]
    kernel = np.ones((2, 2), np.uint8)
    candidates.append(cv2.dilate(digit_norm, kernel, iterations=1))
    candidates.append(cv2.erode(digit_norm, kernel, iterations=1))

    # Ensure all candidates match template size
    resized_candidates: list[np.ndarray] = []
    for cand in candidates:
        if cand.shape != (target_size, target_size):
            cand = cv2.resize(cand, (target_size, target_size), interpolation=cv2.INTER_AREA)
        resized_candidates.append(cand)

    best_digit, best_score = 0, -1.0
    second_best = -1.0
    for candidate in resized_candidates:
        _, cand_bin = cv2.threshold(candidate, 127, 255, cv2.THRESH_BINARY)
        holes = 0
        cnts, hier = cv2.findContours(cand_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            holes = sum(1 for h in hier[0] if h[3] != -1)
        for digit, tmpl_list in templates.items():
            for tmpl in tmpl_list:
                score_ncc = cv2.matchTemplate(candidate, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
                _, tmpl_bin = cv2.threshold(tmpl, 127, 255, cv2.THRESH_BINARY)
                inter = np.logical_and(cand_bin > 0, tmpl_bin > 0).sum()
                union = np.logical_or(cand_bin > 0, tmpl_bin > 0).sum()
                score_iou = inter / union if union > 0 else 0.0
                score = max(float(score_ncc), float(score_iou))

                # Penalize hole mismatch: enforce expected hole counts to separate 6/8/9
                expected_holes = {0: 1, 6: 1, 8: 2, 9: 1}
                if digit in expected_holes:
                    diff = abs(holes - expected_holes[digit])
                    if diff > 0:
                        score *= 0.6 ** diff
                elif digit in {3, 5, 2, 7} and holes > 0:
                    score *= 0.85

                if score > best_score:
                    second_best = best_score
                    best_digit, best_score = digit, float(score)
                elif score > second_best:
                    second_best = float(score)
            # Early exit for this candidate if we hit a very confident match
            if best_score > 0.90 and (best_score - second_best) > 0.08:
                break
        # Early exit overall if confident
        if best_score > 0.90 and (best_score - second_best) > 0.08:
            break

    if debug:
        print(f"[OCR] scores: best={best_score:.3f}, second={second_best:.3f}, sep={best_score-second_best:.3f}, fallback={fallback}")

    thresh = 0.18 if fallback else accept_threshold
    sep = 0.02 if fallback else min_separation
    if best_score < thresh or (best_score - second_best) < sep:
        if fallback and best_digit != 0:
            # In fallback mode, accept the best digit even if scores are low
            if debug:
                print(f"[OCR] fallback accept digit {best_digit} with score {best_score:.3f}")
            return best_digit, best_score
        if debug:
            print(f"[OCR] reject by threshold/sep")
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


def extract_grid_digits(
    binary_grid: np.ndarray,
    templates: dict[int, list[np.ndarray]] | None = None,
    cell_margin: float = 0.0,
    save_cells_dir: str | None = None,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a 9x9 integer grid (0 = empty) from the straightened binary image.

    Args:
        binary_grid: Final binary grid (digits/lines white on black).
        templates: Optional pre-built templates.
        cell_margin: Fraction of a cell to trim on each side to drop grid lines.
        save_cells_dir: If provided, dump the cropped per-cell images for debugging.
        debug: If True, print rejection reasons/score info.

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
    # Ignore a small border inside each cell to avoid grid lines bleeding into OCR
    inner_margin = int(min(cell_h, cell_w) * cell_margin)
    inner_margin = max(0, min(inner_margin, cell_h // 2 - 1, cell_w // 2 - 1))

    if save_cells_dir:
        os.makedirs(save_cells_dir, exist_ok=True)

    board = np.zeros((9, 9), dtype=int)
    scores = np.zeros((9, 9), dtype=float)

    def run_pass(use_fallback: bool, margin: int):
        nonlocal board, scores
        for r in range(9):
            for c in range(9):
                y1, y2 = r * cell_h + margin, (r + 1) * cell_h - margin
                x1, x2 = c * cell_w + margin, (c + 1) * cell_w - margin
                cell1 = digits_only[y1:y2, x1:x2]
                cell2 = raw_view[y1:y2, x1:x2]
                d1, s1 = _extract_digit_from_cell(cell1, templates, fallback=use_fallback, debug=debug)
                d2, s2 = _extract_digit_from_cell(cell2, templates, fallback=use_fallback, debug=debug)
                if s2 > s1:
                    digit, score = d2, s2
                    chosen_cell = cell2
                else:
                    digit, score = d1, s1
                    chosen_cell = cell1
                if board[r, c] == 0 or score > scores[r, c]:
                    board[r, c] = digit
                    scores[r, c] = score
                    if save_cells_dir and debug:
                        suffix = "" if not use_fallback else "_fb"
                        cv2.imwrite(os.path.join(save_cells_dir, f"r{r}_c{c}{suffix}.png"), chosen_cell)

    run_pass(use_fallback=False, margin=inner_margin)

    # If we detected almost nothing, run a permissive fallback pass with smaller margin
    if np.count_nonzero(board) < 5:
        fallback_margin = max(1, int(min(cell_h, cell_w) * 0.03))
        if debug:
            print(f"[OCR] fallback pass, existing hits={np.count_nonzero(board)}")
        run_pass(use_fallback=True, margin=fallback_margin)

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
