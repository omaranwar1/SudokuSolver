"""
Standalone, stricter template-matching OCR for single Sudoku cells.

Designed for high-accuracy matching against the generated digit templates
in `digit_templates/` (50x50, black background, white glyph).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def load_templates(template_dir: str | Path, size: int = 50, allowed_prefixes: Iterable[str] | None = None) -> dict[int, list[np.ndarray]]:
    """Load templates from disk; filter by filename prefix if provided."""
    dir_path = Path(template_dir)
    templates: dict[int, list[np.ndarray]] = {d: [] for d in range(1, 10)}
    if not dir_path.exists():
        return templates

    allowed_lower = [p.lower() for p in allowed_prefixes] if allowed_prefixes else None

    for png in sorted(dir_path.glob("*.png")):
        stem = png.stem
        parts = stem.split("_")
        try:
            digit = int(parts[0])
        except (ValueError, IndexError):
            continue
        if digit not in templates:
            continue
        prefix = "_".join(parts[1:]).lower() if len(parts) > 1 else ""
        if allowed_lower and not any(prefix.startswith(p) for p in allowed_lower):
            continue

        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.mean() > 127:
            img = cv2.bitwise_not(img)
        if img.shape != (size, size):
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        templates[digit].append(img)

    return templates


def _binarize(cell: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=1)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, iterations=1)
    return b


def _isolate_largest_blob(binary: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary
    digit_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.zeros_like(binary)
    mask[labels == digit_label] = 255
    return mask


def normalize_cell(cell: np.ndarray, size: int = 50) -> np.ndarray:
    """Return a normalized, square, binarized 50x50 glyph (white on black)."""
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    work = cell.copy()
    if work.mean() > 127:
        work = cv2.bitwise_not(work)

    b = _binarize(work)

    border = max(2, int(min(b.shape) * 0.08))
    b[:border, :] = 0
    b[-border:, :] = 0
    b[:, :border] = 0
    b[:, -border:] = 0

    b = _isolate_largest_blob(b)

    yx = np.argwhere(b > 0)
    if yx.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    y0, x0 = yx.min(axis=0)
    y1, x1 = yx.max(axis=0)
    roi = b[y0:y1 + 1, x0:x1 + 1]

    side = max(roi.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - roi.shape[0]) // 2
    x_off = (side - roi.shape[1]) // 2
    square[y_off:y_off + roi.shape[0], x_off:x_off + roi.shape[1]] = roi

    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    a_f = (a_f - a_f.mean()) / (a_f.std() + 1e-6)
    b_f = (b_f - b_f.mean()) / (b_f.std() + 1e-6)
    return float(cv2.matchTemplate(a_f, b_f, cv2.TM_CCOEFF_NORMED)[0][0])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = a > 127
    b_bin = b > 127
    inter = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()
    return float(inter / union) if union > 0 else 0.0


def match_cell(
    cell: np.ndarray,
    templates: dict[int, list[np.ndarray]],
    accept_threshold: float = 0.6,
    min_gap: float = 0.08,
) -> dict[str, object]:
    """
    Match a normalized cell against templates.

    Returns: dict with digit, score, gap, per_digit scores.
    """
    norm = normalize_cell(cell, size=next(iter(templates.values()))[0].shape[0])
    candidates = [
        norm,
        cv2.bitwise_not(norm),
    ]

    per_digit = {d: -1.0 for d in range(1, 10)}
    best_digit, best_score, second_best = 0, -1.0, -1.0
    for cand in candidates:
        for digit, tmpl_list in templates.items():
            for tmpl in tmpl_list:
                score = 0.7 * _ncc(cand, tmpl) + 0.3 * _iou(cand, tmpl)
                per_digit[digit] = max(per_digit[digit], score)
                if score > best_score:
                    second_best = best_score
                    best_digit, best_score = digit, score
                elif score > second_best:
                    second_best = score

    gap = best_score - second_best
    if best_score < accept_threshold or gap < min_gap:
        best_digit = 0

    return {
        "digit": best_digit,
        "score": best_score,
        "gap": gap,
        "per_digit": per_digit,
        "normalized": norm,
    }
