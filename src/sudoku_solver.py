"""
Sudoku Solver - Main Application Module

This module orchestrates the complete Sudoku solving pipeline:
1. Image preprocessing
2. Grid detection and isolation
3. Perspective transformation
4. (Future: OCR and puzzle solving - Milestone 2)

Usage:
    python -m src.sudoku_solver --image path/to/image.jpg

Author: CSCE 4603 Project Team
"""

import cv2
import numpy as np
import argparse
import os
import sys

from .preprocessing import preprocess_image, resize_image
from .grid_detection import (
    find_largest_contour,
    find_grid_corners,
    draw_contour_and_corners,
    validate_grid_is_square,
    draw_9x9_grid_lines,
    detect_grid_using_lines,
    draw_template_grid,
    reinforce_grid_morphological,
    reinforce_grid_hough,
    reinforce_grid_adaptive,
    detect_blocked_corners
)
from .perspective_transform import perspective_transform, get_transform_quality_score


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

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Load and resize image
        print("\n[1/5] Loading image...")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not load image from {image_path}")

        original = resize_image(original, max_width=800)
        self.intermediate_images['original'] = original
        print(f"      Image size: {original.shape[1]}x{original.shape[0]}")

        # Step 2: Preprocess using robust preprocessing pipeline
        print("\n[2/5] Preprocessing image...")
        print("      Using advanced preprocessing pipeline:")
        print("      - Noise removal (adaptive)")
        print("      - Brightness normalization")
        print("      - Adaptive thresholding")
        print("      - Grid line completion")
        print("      - Rotation correction (if needed)")

        # Use the robust preprocessing pipeline (pass the resized image)
        # Returns: (original, enhanced_grayscale, binary)
        _, enhanced_gray, processed = preprocess_image(original, display=False)

        self.intermediate_images['preprocessed'] = processed
        self.intermediate_images['enhanced_gray'] = enhanced_gray

        # Step 3: Detect grid - ADAPTIVE STRATEGY: Try BOTH methods and select the best
        print("\n[3/5] Detecting 9x9 grid structure...")
        print(f"      Using preprocessed binary image: {processed.shape}")
        print(f"      Preprocessed type: {processed.dtype}, min={processed.min()}, max={processed.max()}")
        print("      Strategy: Adaptive detection - evaluating both contour and line-based methods")

        # Try BOTH detection methods
        contour_result = None
        line_result = None

        # Method 1: Contour detection (best for clear, strong grid borders)
        print("\n      [Method 1] Contour detection...")
        contour = find_largest_contour(processed)

        if contour is not None:
            area = cv2.contourArea(contour)
            print(f"        ✓ Contour found (area: {area:.0f} pixels)")

            corners_contour = find_grid_corners(contour)
            quality_contour = get_transform_quality_score(corners_contour)
            is_square_c, aspect_ratio_c, _ = validate_grid_is_square(corners_contour)

            # Calculate grid coverage (what % of image does grid occupy)
            image_area = processed.shape[0] * processed.shape[1]
            coverage_contour = area / image_area

            print(f"        Quality: {quality_contour:.3f}, Ratio: {aspect_ratio_c:.3f}, Coverage: {coverage_contour:.3f}")

            contour_result = {
                'corners': corners_contour,
                'quality': quality_contour,
                'aspect_ratio': aspect_ratio_c,
                'coverage': coverage_contour,
                'contour': contour
            }
        else:
            print(f"        ✗ No contour found")

        # Method 2: Line-based detection (robust for faint/noisy grids)
        print("\n      [Method 2] Line-based detection (Hough Transform)...")
        corners_lines = detect_grid_using_lines(processed, debug=False)

        if corners_lines is not None:
            quality_lines = get_transform_quality_score(corners_lines)
            is_square_l, aspect_ratio_l, _ = validate_grid_is_square(corners_lines)

            # Estimate coverage from corner positions
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

        # Select the best method using adaptive scoring
        corners = None
        detection_method = None

        if contour_result is None and line_result is None:
            print("\n      ERROR: Both detection methods failed!")
            print("      Saving failed preprocessed image for debugging...")
            cv2.imwrite('debug_preprocessed_failed.jpg', processed)
            return None

        # If only one method succeeded, use it
        if contour_result is None:
            corners = line_result['corners']
            detection_method = "line-based"
            contour = None
            print(f"\n      ✓ Selected: Line-based (only method that succeeded)")
        elif line_result is None:
            corners = contour_result['corners']
            detection_method = "contour-based"
            contour = contour_result['contour']
            print(f"\n      ✓ Selected: Contour-based (only method that succeeded)")
        else:
            # Both methods succeeded - compare and select the best
            print(f"\n      [Adaptive Selection] Comparing methods...")

            # Calculate composite scores for each method (used for bias gate)
            def calculate_score(result):
                quality_score = result['quality']  # 0-1, higher is better
                ratio_score = 1.0 - abs(1.0 - result['aspect_ratio'])  # 0-1, closer to 1.0 is better
                ratio_score = max(0, ratio_score)  # Clamp to 0-1
                coverage_score = min(result['coverage'], 1.0)  # 0-1, higher is better (but cap at 1.0)

                composite = (quality_score * 0.5 +
                           ratio_score * 0.3 +
                           coverage_score * 0.2)
                return composite

            score_contour = calculate_score(contour_result)
            score_lines = calculate_score(line_result)

            print(f"        Contour score: {score_contour:.3f} (q={contour_result['quality']:.3f}, r={contour_result['aspect_ratio']:.3f}, c={contour_result['coverage']:.3f})")
            print(f"        Line score:    {score_lines:.3f} (q={line_result['quality']:.3f}, r={line_result['aspect_ratio']:.3f}, c={line_result['coverage']:.3f})")

            # Hierarchy:
            # 1) Bias gate: contour must pass the +0.075 bias; otherwise never selected.
            score_diff = score_lines - score_contour
            BIAS_THRESHOLD = 0.075

            if score_diff > BIAS_THRESHOLD:
                corners = line_result['corners']
                detection_method = "line-based"
                contour = None
                print(f"      ✓ Selected: Line-based (score {score_lines:.3f} > {score_contour:.3f} + {BIAS_THRESHOLD})")
            else:
                # 2) Check for blocked corners; if blocked, apply coverage rule.
                is_blocked, block_reason = detect_blocked_corners(
                    contour_result['corners'],
                    line_result['corners'],
                    processed,
                    debug=True
                )

                if not is_blocked:
                    corners = contour_result['corners']
                    detection_method = "contour-based"
                    contour = contour_result['contour']
                    print(f"      ✓ Selected: Contour-based (passes bias; no blockage detected)")
                else:
                    print(f"\n      ! Corner displacement detected: {block_reason}")
                    coverage_diff = line_result['coverage'] - contour_result['coverage']
                    COVERAGE_THRESHOLD = 0.20

                    if coverage_diff < COVERAGE_THRESHOLD:
                        corners = line_result['corners']
                        detection_method = "line-based"
                        contour = None
                        print(f"      ✓ Selected: Line-based (blocked corner + small coverage gap {coverage_diff:.3f} < {COVERAGE_THRESHOLD})")
                    else:
                        corners = contour_result['corners']
                        detection_method = "contour-based"
                        contour = contour_result['contour']
                        print(f"      ✓ Selected: Contour-based (blocked corner + large coverage gap {coverage_diff:.3f} >= {COVERAGE_THRESHOLD})")

        print("\n[4/5] Corners identified...")
        print(f"      Detection method: {detection_method}")
        print(f"      Corners detected:")
        print(f"        Top-Left:     ({corners[0][0]:.1f}, {corners[0][1]:.1f})")
        print(f"        Top-Right:    ({corners[1][0]:.1f}, {corners[1][1]:.1f})")
        print(f"        Bottom-Right: ({corners[2][0]:.1f}, {corners[2][1]:.1f})")
        print(f"        Bottom-Left:  ({corners[3][0]:.1f}, {corners[3][1]:.1f})")

        # Calculate quality score
        quality = get_transform_quality_score(corners)
        print(f"      Quality score: {quality:.2f}")

        # Validate that grid is square (9x9 Sudoku should be square)
        is_square, aspect_ratio, message = validate_grid_is_square(corners)
        print(f"      {message}")
        if not is_square:
            print(f"      WARNING: Detected grid may not be a proper Sudoku grid!")

        # Draw detection visualization on PREPROCESSED BINARY (to show what detection sees)
        processed_3ch = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # If we have a contour, draw it; otherwise just draw corners
        if contour is not None:
            contour_viz = draw_contour_and_corners(processed_3ch, contour, corners)
        else:
            # No contour (line-based detection), just draw corners and grid outline
            contour_viz = processed_3ch.copy()
            # Draw corners as red circles
            for i, corner in enumerate(corners):
                x, y = int(corner[0]), int(corner[1])
                cv2.circle(contour_viz, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(contour_viz, str(i), (x + 15, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # Draw grid outline in green
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(contour_viz, [pts], True, (0, 255, 0), 3)

        self.intermediate_images['contour_detection'] = contour_viz

        # Draw 9x9 grid lines to visualize Sudoku cell structure
        grid_viz = draw_9x9_grid_lines(processed_3ch.copy(), corners, self.output_size)
        self.intermediate_images['grid_9x9_visualization'] = grid_viz

        # Convert enhanced grayscale to 3-channel for straightening
        enhanced_gray_3ch = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # Step 5: Apply perspective transformation
        print("\n[5/5] Applying perspective transformation...")

        # Use enhanced grayscale for straightened output (better for OCR)
        straightened = perspective_transform(enhanced_gray_3ch, corners, self.output_size)
        self.intermediate_images['straightened'] = straightened
        print(f"      Grid straightened to {self.output_size}x{self.output_size}")
        print(f"      Using brightness-enhanced image for straightened output")

        # Apply to binary image as well for future OCR
        straightened_binary = perspective_transform(processed, corners, self.output_size)
        self.intermediate_images['straightened_binary'] = straightened_binary

        # Use adaptive method - detects actual line positions and only reinforces if broken
        print("      Analyzing grid completeness...")

        # This function:
        # 1. Detects actual lines (even if angled)
        # 2. Counts them
        # 3. Only applies morphological reinforcement if < 12 lines detected
        final_binary = reinforce_grid_adaptive(straightened_binary, self.output_size)

        # Check if reinforcement was applied
        if not np.array_equal(final_binary, straightened_binary):
            self.intermediate_images['straightened_binary_reinforced'] = final_binary
            print(f"      ✓ Grid reinforcement applied")
        else:
            print(f"      ✓ Grid is complete, no reinforcement needed")

        # Save intermediate results
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

        # Save each intermediate image
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

    args = parser.parse_args()

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

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
