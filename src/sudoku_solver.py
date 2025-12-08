"""
Sudoku Solver - Main Application Module
"""

import cv2
import numpy as np
import argparse
import os
import sys

from .preprocessing import preprocess_image, resize_image
from .grid_detection import (
    draw_contour_and_corners,
    validate_grid_is_square,
    draw_9x9_grid_lines,
    detect_grid_hybrid,
    detect_grid_using_lines,
    reinforce_grid_adaptive,
)
from .perspective_transform import perspective_transform, get_transform_quality_score
from .ocr import extract_grid_digits, format_board, render_solution_on_image, resolve_conflicts
from .solver import solve_puzzle


class SudokuSolver:
    """
    Main class for the Sudoku Solver application.

    This class encapsulates the entire pipeline for processing a Sudoku image
    and extracting the grid in a normalized, straightened format.
    """

    def __init__(self, output_size=450, save_intermediate=True):
        """
        Initialize the Sudoku Solver.

        Args:
            output_size (int): Size of the output straightened grid (default: 450x450)
            save_intermediate (bool): Whether to save intermediate processing steps
        """
        self.output_size = output_size
        self.save_intermediate = save_intermediate
        self.intermediate_images = {}

    def process_image(self, image_path, output_dir='output'):
        """
        Process a Sudoku image through the complete Milestone 1 pipeline.

        Pipeline steps:
        1. Load and resize image
        2. Preprocess (grayscale, blur, threshold, morphology)
        3. Detect grid contour
        4. Identify four corners
        5. Apply perspective transformation

        Args:
            image_path (str): Path to the input image
            output_dir (str): Directory to save output images

        Returns:
            dict: Results containing processed images and metadata
        """
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        os.makedirs(output_dir, exist_ok=True)

        print("\n[1/5] Loading image...")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not load image from {image_path}")

        original = resize_image(original, max_width=800)
        self.intermediate_images['original'] = original
        print(f"      Image size: {original.shape[1]}x{original.shape[0]}")
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        print("\n[2/5] Preprocessing image...")
        print("      Using advanced preprocessing pipeline:")
        print("      - Noise removal (adaptive)")
        print("      - Brightness normalization")
        print("      - Adaptive thresholding")
        print("      - Grid line completion")
        print("      - Rotation correction (if needed)")

        _, enhanced_gray, processed = preprocess_image(original, display=False)

        self.intermediate_images['preprocessed'] = processed
        self.intermediate_images['enhanced_gray'] = enhanced_gray

        print("\n[3/5] Detecting 9x9 grid structure...")
        print(f"      Using preprocessed binary image: {processed.shape}")
        print(f"      Preprocessed type: {processed.dtype}, min={processed.min()}, max={processed.max()}")
        print("      Strategy: Adaptive detection - evaluating both contour and line-based methods")

        contour_result = None
        line_result = None

        print("\n      [Method 1] Hybrid detection (minAreaRect + Hough fallback)...")
        corners_hybrid, contour = detect_grid_hybrid(processed, debug=False)

        if corners_hybrid is not None:
            if contour is not None:
                area = cv2.contourArea(contour)
                image_area = processed.shape[0] * processed.shape[1]
                coverage_contour = area / image_area
            else:
                area = 0
                coverage_contour = 0

            quality_contour = get_transform_quality_score(corners_hybrid)
            is_square_c, aspect_ratio_c, _ = validate_grid_is_square(corners_hybrid)

            print(f"        ✓ Hybrid corners found (area: {area:.0f} pixels)")
            print(f"        Quality: {quality_contour:.3f}, Ratio: {aspect_ratio_c:.3f}, Coverage: {coverage_contour:.3f}")

            contour_result = {
                'corners': corners_hybrid,
                'quality': quality_contour,
                'aspect_ratio': aspect_ratio_c,
                'coverage': coverage_contour,
                'contour': contour
            }
        else:
            print(f"        ✗ Hybrid detection failed")

        print("\n      [Method 2] Line-based detection (Hough Transform)...")
        corners_lines = detect_grid_using_lines(processed, debug=False)

        if corners_lines is not None:
            quality_lines = get_transform_quality_score(corners_lines)
            is_square_l, aspect_ratio_l, _ = validate_grid_is_square(corners_lines)

            width = max(corners_lines[:, 0]) - min(corners_lines[:, 0])
            height = max(corners_lines[:, 1]) - min(corners_lines[:, 1])
            grid_area = width * height
            coverage_lines = grid_area / (processed.shape[0] * processed.shape[1])

            print(f"        ✓ Lines detected")
            print(f"        Quality: {quality_lines:.3f}, Ratio: {aspect_ratio_l:.3f}, Coverage: {coverage_lines:.3f}")

            line_result = {
                'corners': corners_lines,
                'quality': quality_lines,
                'aspect_ratio': aspect_ratio_l,
                'coverage': coverage_lines
            }
        else:
            print(f"        ✗ Line detection failed")

        corners = None
        detection_method = None

        if contour_result is None and line_result is None:
            print("\n      ERROR: Both detection methods failed!")
            print("      Saving failed preprocessed image for debugging...")
            cv2.imwrite('debug_preprocessed_failed.jpg', processed)
            return None

        # Prefer contour-based corners for OCR/perspective; fall back to lines only if contour fails.
        if contour_result is not None:
            corners = contour_result['corners']
            detection_method = "contour-based (preferred)"
            contour = contour_result['contour']
            print(f"\n      ✓ Selected: Contour-based (preferred)")
        else:
            corners = line_result['corners']
            detection_method = "line-based (fallback)"
            contour = None
            print(f"\n      ✓ Selected: Line-based (contour unavailable)")

        print("\n[4/5] Corners identified...")
        print(f"      Detection method: {detection_method}")
        print(f"      Corners detected:")
        print(f"        Top-Left:     ({corners[0][0]:.1f}, {corners[0][1]:.1f})")
        print(f"        Top-Right:    ({corners[1][0]:.1f}, {corners[1][1]:.1f})")
        print(f"        Bottom-Right: ({corners[2][0]:.1f}, {corners[2][1]:.1f})")
        print(f"        Bottom-Left:  ({corners[3][0]:.1f}, {corners[3][1]:.1f})")

        quality = get_transform_quality_score(corners)
        print(f"      Quality score: {quality:.2f}")

        is_square, aspect_ratio, message = validate_grid_is_square(corners)
        print(f"      {message}")
        if not is_square:
            print(f"      WARNING: Detected grid may not be a proper Sudoku grid!")

        processed_3ch = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        if contour is not None:
            contour_viz = draw_contour_and_corners(processed_3ch, contour, corners)
        else:
            contour_viz = processed_3ch.copy()
            for i, corner in enumerate(corners):
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(contour_viz, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(contour_viz, str(i), (x + 15, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(contour_viz, [pts], True, (0, 255, 0), 3)

        self.intermediate_images['contour_detection'] = contour_viz

        grid_viz = draw_9x9_grid_lines(processed_3ch.copy(), corners, self.output_size)
        self.intermediate_images['grid_9x9_visualization'] = grid_viz

        enhanced_gray_3ch = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        print("\n[5/5] Applying perspective transformation...")

        straightened = perspective_transform(enhanced_gray_3ch, corners, self.output_size)
        self.intermediate_images['straightened'] = straightened
        print(f"      Grid straightened to {self.output_size}x{self.output_size}")
        print(f"      Using brightness-enhanced image for straightened output")

        straightened_binary = perspective_transform(processed, corners, self.output_size)
        self.intermediate_images['straightened_binary'] = straightened_binary

        print("      Analyzing grid completeness...")

        final_binary = reinforce_grid_adaptive(straightened_binary, self.output_size)

        if not np.array_equal(final_binary, straightened_binary):
            self.intermediate_images['straightened_binary_reinforced'] = final_binary
            print(f"      ✓ Grid reinforcement applied")
        else:
            print(f"      ✓ Grid is complete, no reinforcement needed")

        print("\n[6/6] OCR (pattern matching) and solving...")
        cells_dir = None
        if self.save_intermediate:
            cells_dir = os.path.join(output_dir, f"{base_name}_cells")
        # Contour-first OCR
        board_raw_c, scores_c = extract_grid_digits(final_binary, save_cells_dir=cells_dir)
        board_c, conflicts_c = resolve_conflicts(board_raw_c, scores_c)
        contour_candidate = {
            "board": board_c,
            "conflicts": conflicts_c,
            "scores": scores_c,
            "final_binary": final_binary,
            "straightened": straightened,
            "label": "contour-based",
        }

        # If contour OCR has conflicts, try line-based OCR and pick the one with fewer conflicts.
        line_candidate = None
        if line_result is not None and contour_candidate["conflicts"]:
            print("\n      Contour OCR has conflicts; trying line-based corners for comparison...")
            straightened_line = perspective_transform(enhanced_gray_3ch, line_result['corners'], self.output_size)
            straightened_binary_line = perspective_transform(processed, line_result['corners'], self.output_size)
            final_binary_line = reinforce_grid_adaptive(straightened_binary_line, self.output_size)

            if self.save_intermediate:
                self.intermediate_images['straightened_line'] = straightened_line
                self.intermediate_images['straightened_binary_line'] = straightened_binary_line
                self.intermediate_images['straightened_binary_line_reinforced'] = final_binary_line

            cells_dir_line = None
            if self.save_intermediate:
                cells_dir_line = os.path.join(output_dir, f"{base_name}_cells_line")

            board_raw_line, scores_line = extract_grid_digits(final_binary_line, save_cells_dir=cells_dir_line)
            board_line, conflicts_line = resolve_conflicts(board_raw_line, scores_line)
            line_candidate = {
                "board": board_line,
                "conflicts": conflicts_line,
                "scores": scores_line,
                "final_binary": final_binary_line,
                "straightened": straightened_line,
                "label": "line-based",
            }

        # Choose candidate with fewest conflicts (tie-break: more givens). Solve only after selection.
        candidates = [contour_candidate] + ([line_candidate] if line_candidate is not None else [])
        candidates = [c for c in candidates if c is not None]
        candidates.sort(key=lambda c: (len(c["conflicts"]), -np.count_nonzero(c["board"])))
        chosen = candidates[0]
        board = chosen["board"]
        conflict_notes = chosen["conflicts"]
        final_binary = chosen["final_binary"]
        chosen_straightened = chosen["straightened"]
        if chosen["label"] != "contour-based":
            detection_method = f"{chosen['label']} (fallback due to conflicts)"

        print(f"\n      Detected puzzle ({chosen['label']}):")
        print(format_board(board))
        if conflict_notes:
            print("      Note: cleaned duplicate detections:")
            for note in conflict_notes:
                print(f"        - {note}")

        given_count = np.count_nonzero(board)
        if given_count < 15:
            print(f"      WARNING: Low number of detected givens ({given_count}); OCR may be unreliable")

        solution, solve_msg = solve_puzzle(board)
        if solution is None:
            print(f"      ✗ Could not solve: {solve_msg}")
            print("        (Check OCR accuracy or image quality)")
        else:
            print(f"      ✓ Solved puzzle ({solve_msg}):")
            print(format_board(solution))
            solved_overlay = render_solution_on_image(
                chosen_straightened,
                solution,
                board
            )
            self.intermediate_images['solved_overlay'] = solved_overlay

        if self.save_intermediate:
            self._save_results(image_path, output_dir)

        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Results saved to: {output_dir}/")
        print(f"{'='*60}\n")

        return {
            'original': original,
            'preprocessed': processed,
            'contour': contour,
            'corners': corners,
            'straightened': straightened,
            'straightened_binary': final_binary,
            'quality_score': quality
        }

    def _save_results(self, image_path, output_dir):
        """
        Save intermediate processing results to disk.

        Args:
            image_path (str): Original image path (for naming)
            output_dir (str): Output directory
        """
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for name, image in self.intermediate_images.items():
            output_path = os.path.join(output_dir, f"{base_name}_{name}.jpg")
            cv2.imwrite(output_path, image)

        print(f"\n      Saved {len(self.intermediate_images)} intermediate images")


def main():
    """
    Main entry point for the Sudoku Solver application.

    Handles command-line arguments and processes images.
    """
    parser = argparse.ArgumentParser(
        description='Sudoku Solver - Computer Vision Project (Milestone 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a single image:
    python -m src.sudoku_solver --image 01.jpg

  Process with custom output size:
    python -m src.sudoku_solver --image 01.jpg --size 600

  Process all test images:
    python -m src.sudoku_solver --image *.jpg
        """
    )

    parser.add_argument('--image', '-i', required=True,
                        help='Path to input Sudoku image')
    parser.add_argument('--output', '-o', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--size', '-s', type=int, default=450,
                        help='Output grid size in pixels (default: 450)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save intermediate images')
    parser.add_argument('--binary-grid', action='store_true',
                        help='Treat input image as an already straightened binary grid (skip detection)')

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # If user provides a pre-straightened binary grid, skip the full pipeline
    if args.binary_grid:
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not load binary grid from {args.image}")
            sys.exit(1)
        # Ensure binary polarity: digits white on black
        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if bin_img.mean() > 127:
            bin_img = cv2.bitwise_not(bin_img)
        os.makedirs(args.output, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        cells_dir = os.path.join(args.output, f"{base}_cells")
        board_raw, scores = extract_grid_digits(bin_img, save_cells_dir=cells_dir)
        board, conflict_notes = resolve_conflicts(board_raw, scores)
        print("\nDetected puzzle (binary input):")
        print(format_board(board))
        if conflict_notes:
            print("Note: cleaned duplicate detections:")
            for note in conflict_notes:
                print(f"  - {note}")
        sys.exit(0)

    # Create solver instance
    solver = SudokuSolver(
        output_size=args.size,
        save_intermediate=not args.no_save
    )

    # Process the image
    try:
        result = solver.process_image(args.image, args.output)

        if result is None:
            print("\nProcessing failed. Please check the image and try again.")
            sys.exit(1)

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


def check_if_grid_is_skewed(binary_image: np.ndarray) -> tuple[bool, str]:
    """
    Check if grid is badly skewed/rotated by detecting diagonal lines.

    Returns:
        (is_skewed, reason)
    """
    edges = cv2.Canny(binary_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=40, maxLineGap=15)

    if lines is None or len(lines) < 5:
        return False, "Not enough lines to check skew"

    # Calculate angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        # Normalize to 0-90
        if angle > 90:
            angle = 180 - angle
        angles.append(angle)

    # Count orthogonal vs diagonal lines
    orthogonal = sum(1 for a in angles if a < 15 or a > 75)  # horizontal/vertical ±15°
    diagonal = len(angles) - orthogonal

    if len(angles) > 0:
        diagonal_ratio = diagonal / len(angles)
        if diagonal_ratio > 0.3:  # More than 30% diagonal
            return True, f"Grid is skewed ({diagonal_ratio:.0%} diagonal lines detected)"

    return False, "Grid alignment OK"


def needs_grid_overlay(binary_image: np.ndarray) -> tuple[bool, str]:
    """
    Check if grid has missing lines and needs reinforcement.

    Use Hough Line Transform to detect actual straight lines.

    Returns:
        (needs_filling, reason)
    """
    h, w = binary_image.shape

    # Use Hough Line Transform to detect straight lines
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                           minLineLength=int(min(h, w) * 0.4), maxLineGap=30)

    if lines is None:
        print(f"      Grid line detection:")
        print(f"        No lines detected with Hough Transform")
        print(f"      → Grid broken: no detectable lines")
        return True, "Grid broken: no detectable lines"

    # Separate detected lines into horizontal and vertical
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # Horizontal lines (angle close to 0 or 180)
        if angle < 20 or angle > 160:
            horizontal_lines.append((x1, y1, x2, y2, angle))
        # Vertical lines (angle close to 90)
        elif 70 < angle < 110:
            vertical_lines.append((x1, y1, x2, y2, angle))

    h_count = len(horizontal_lines)
    v_count = len(vertical_lines)
    total_detected = h_count + v_count

    print(f"      Grid line detection (Hough Transform):")
    print(f"        Horizontal lines detected: {h_count}")
    print(f"        Vertical lines detected: {v_count}")
    print(f"        Total lines detected: {total_detected}")

    # SIMPLE DECISION:
    # A complete 9x9 grid should have ~20 lines (10 horizontal + 10 vertical)
    # If we detect fewer than 12 lines, the grid is broken

    MIN_LINES_THRESHOLD = 12

    if total_detected < MIN_LINES_THRESHOLD:
        reason = f"Grid broken: only {total_detected} lines detected (< {MIN_LINES_THRESHOLD})"
        print(f"      → {reason}")
        return True, reason
    else:
        reason = f"Grid complete: {total_detected} lines detected (>= {MIN_LINES_THRESHOLD})"
        print(f"      → {reason}")
        return False, reason
