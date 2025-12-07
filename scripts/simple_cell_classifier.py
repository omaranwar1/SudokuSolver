"""
Minimal cell classifier: given a single cell image PNG, decide its digit (1-9) or empty.

Usage:
  python scripts/simple_cell_classifier.py output_debug/01_cells/r0_c0.png
  python scripts/simple_cell_classifier.py some_cell.png --templates digit_templates --accept 0.55 --gap 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


def load_templates(template_dir: Path, size: int = 50) -> Dict[int, List[np.ndarray]]:
    """Load templates from disk. Expects files named like `1_*.png`."""
    templates: Dict[int, List[np.ndarray]] = {d: [] for d in range(1, 10)}
    for png in sorted(template_dir.glob("*.png")):
        stem = png.stem
        try:
            digit = int(stem.split("_")[0])
        except (ValueError, IndexError):
            continue
        if digit not in templates:
            continue
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Templates are assumed correct as-is (already black background, white digit, correct size)
        if img.shape != (size, size):
            raise ValueError(f"Template {png} has shape {img.shape}, expected {(size, size)}")
        templates[digit].append(img)
    return templates


def overlap_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Binary overlap with penalty for extra ink:
    score = intersection / (template_fg + extra_candidate_fg)
    """
    a_bin = a > 127
    b_bin = b > 127
    inter = np.logical_and(a_bin, b_bin).sum()
    tmpl_fg = b_bin.sum()
    cand_fg = a_bin.sum()
    extra = max(cand_fg - inter, 0)
    denom = tmpl_fg + extra
    return float(inter / denom) if denom > 0 else 0.0


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift image by (dx, dy) with zero fill."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)


def scale_image_to_fit(img: np.ndarray, target_size: int, delta: int) -> np.ndarray:
    """
    Resize image by delta pixels (relative to target) while keeping canvas size.
    Positive delta upsizes, negative downsizes.
    """
    new_size = max(10, target_size + delta)
    scaled = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = max(0, (target_size - new_size) // 2)
    x_off = max(0, (target_size - new_size) // 2)
    y1 = min(target_size, y_off + scaled.shape[0])
    x1 = min(target_size, x_off + scaled.shape[1])
    canvas[y_off:y1, x_off:x1] = scaled[: y1 - y_off, : x1 - x_off]
    return canvas


def save_overlap_debug(candidate: np.ndarray, tmpl: np.ndarray, out_path: Path) -> None:
    """
    Save a debug image showing overlap: white = overlap, gray = template-only, black = background/candidate-only.
    """
    cand_bin = candidate > 127
    tmpl_bin = tmpl > 127
    overlap = np.logical_and(cand_bin, tmpl_bin)
    tmpl_only = np.logical_and(tmpl_bin, ~cand_bin)
    debug = np.zeros(candidate.shape, dtype=np.uint8)
    debug[tmpl_only] = 128
    debug[overlap] = 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), debug)


def preprocess_with_steps(cell: np.ndarray, target_size: int, border_frac: float = 0.08) -> Dict[str, np.ndarray]:
    """
    Apply minimal preprocessing and return intermediate steps for visualization.
    Steps:
      - polarity correction (digit white on black)
      - binarize + clear borders
    """
    steps: Dict[str, np.ndarray] = {}

    if cell.shape != (target_size, target_size):
        cell = cv2.resize(cell, (target_size, target_size), interpolation=cv2.INTER_AREA)
    steps["original"] = cell

    # Polarity
    work = cell.copy()
    if work.mean() > 127:
        work = cv2.bitwise_not(work)
    steps["polarity_fixed"] = work

    # Binarize and clear borders
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = max(1, int(min(binary.shape) * border_frac))
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0
    steps["border_cleared"] = binary

    return steps


def classify_cell(
    cell_path: Path,
    templates: Dict[int, List[np.ndarray]],
    empty_thresh: float,
    dump_dir: Path | None = None,
    debug_digit: int | None = None,
) -> None:
    cell = cv2.imread(str(cell_path), cv2.IMREAD_GRAYSCALE)
    if cell is None:
        print(f"{cell_path.name}: could not load")
        return
    target_size = next(iter(templates.values()))[0].shape[0]

    steps = preprocess_with_steps(cell, target_size=target_size)
    base = steps["border_cleared"]

    # Empty check: if very little ink or tiny bbox, treat as empty and skip digit matching
    ink_fraction = np.count_nonzero(base) / base.size
    nz = np.argwhere(base > 0)
    bbox_area = 0
    if nz.size > 0:
        y0, x0 = nz.min(axis=0)
        y1, x1 = nz.max(axis=0)
        bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)

    min_bbox_area = 0.02 * base.size

    if ink_fraction < empty_thresh or (bbox_area > 0 and bbox_area < min_bbox_area):
        if dump_dir:
            dump_dir.mkdir(parents=True, exist_ok=True)
            for name, img in steps.items():
                cv2.imwrite(str(dump_dir / f"{cell_path.stem}_{name}.png"), img)
        print(f"{cell_path.name}: empty (ink={ink_fraction:.4f}, bbox={bbox_area/base.size:.4f})")
        return

    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
        for name, img in steps.items():
            cv2.imwrite(str(dump_dir / f"{cell_path.stem}_{name}.png"), img)

    # Generate candidate variants with small shifts and scales to improve alignment.
    target_size = base.shape[0]
    base_variants: list[np.ndarray] = []
    for delta in (-2, 0, 2):
        scaled = scale_image_to_fit(base, target_size, delta)
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                base_variants.append(shift_image(scaled, dx, dy))

    candidates = []
    for var in base_variants:
        candidates.append(var)
        candidates.append(cv2.bitwise_not(var))

    per_digit = {d: -1.0 for d in range(1, 10)}
    best_digit, best_score, second_best = 0, -1.0, -1.0
    for cand in candidates:
        for digit, tmpl_list in templates.items():
            for tmpl in tmpl_list:
                score = overlap_score(cand, tmpl)
                per_digit[digit] = max(per_digit[digit], score)
                if score > best_score:
                    second_best = best_score
                    best_digit, best_score = digit, score
                elif score > second_best:
                    second_best = score
                if dump_dir and debug_digit and digit == debug_digit:
                    save_overlap_debug(cand, tmpl, dump_dir / f"{cell_path.stem}_overlap_{digit}.png")

    margin = best_score - second_best
    per_digit_str = " ".join(f"{d}:{per_digit[d]:.2f}" for d in range(1, 10))
    print(f"{cell_path.name}: digit={best_digit} (score={best_score:.3f}, margin={margin:.3f}) | {per_digit_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a single Sudoku cell using template matching.")
    parser.add_argument("cell", type=Path, help="Path to cell PNG.")
    parser.add_argument("--templates", type=Path, default=Path("digit_templates"), help="Directory of templates.")
    parser.add_argument("--empty-thresh", type=float, default=0.01, help="Ink fraction below which the cell is empty.")
    parser.add_argument("--dump-dir", type=Path, help="Optional directory to dump intermediate steps as PNGs.")
    parser.add_argument("--debug-digit", type=int, help="If set, save overlap debug image for this digit.")
    args = parser.parse_args()

    templates = load_templates(args.templates)
    classify_cell(
        args.cell,
        templates,
        empty_thresh=args.empty_thresh,
        dump_dir=args.dump_dir,
        debug_digit=args.debug_digit,
    )


if __name__ == "__main__":
    main()
