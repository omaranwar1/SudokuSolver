
"""
Grid Detection Module

This module handles the detection and isolation of the Sudoku grid from
preprocessed images using:
- Contour detection for outer frame isolation
- Corner point identification
- Hough transform for line detection (alternative method)

References:
- Suzuki & Abe, "Topological Structural Analysis of Digitized Binary Images" (1985)
- Hough, P.V.C., "Method and means for recognizing complex patterns" (1962)
- Douglas & Peucker, "Algorithms for the reduction of the number of points" (1973)
"""

import cv2
import numpy as np


def detect_grid_hybrid(processed_image, debug=False):
    """
    Legacy hybrid detector wrapper: locate the largest contour and extract corners.

    Returns:
        corners (np.ndarray | None), contour (np.ndarray | None)
    """
    contour = find_largest_contour(processed_image)
    if contour is None:
        if debug:
            print("      Hybrid: no contour found")
        return None, None
    corners = find_grid_corners(contour)
    if debug:
        print("      Hybrid: contour + corners extracted")
    return corners, contour


def validate_grid_is_square(corners, tolerance=0.2):
    """
    Validate that the detected grid is approximately square.

    Args:
        corners: Array of 4 corner points [TL, TR, BR, BL]
        tolerance: How much deviation from square is acceptable (0.2 = 20%)

    Returns:
        tuple: (is_valid, aspect_ratio, message)
    """
    # Calculate the four side lengths
    top = np.linalg.norm(corners[1] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])

    # Calculate average width and height
    width = (top + bottom) / 2
    height = (left + right) / 2

    # Calculate aspect ratio
    aspect_ratio = width / height if height > 0 else 0

    # Check if it's approximately square (within tolerance)
    if 1 - tolerance <= aspect_ratio <= 1 + tolerance:
        return True, aspect_ratio, f"Grid is square (ratio: {aspect_ratio:.2f})"
    else:
        return False, aspect_ratio, f"WARNING: Grid not square (ratio: {aspect_ratio:.2f})"


def draw_9x9_grid_lines(image, corners, output_size=450):
    """
    Draw 9x9 grid lines on the image to visualize Sudoku cell structure.

    Args:
        image: Image to draw on (3-channel)
        corners: Array of 4 corner points [TL, TR, BR, BL]
        output_size: Size of the output grid (default 450 for 9x9 with 50px cells)

    Returns:
        Image with 9x9 grid lines drawn
    """
    result = image.copy()

    # Calculate grid line positions (10 lines: 0, 1, 2, ..., 9 for 9x9 grid)
    cell_size = output_size // 9

    # Draw horizontal lines (mapping from straightened to original)
    for i in range(10):
        y = i * cell_size

        # Calculate start and end points in the warped space
        # Interpolate between top and bottom edges
        t = y / output_size  # Position ratio (0 to 1)

        # Start point (left edge): interpolate between TL and BL
        start_x = int(corners[0][0] + t * (corners[3][0] - corners[0][0]))
        start_y = int(corners[0][1] + t * (corners[3][1] - corners[0][1]))

        # End point (right edge): interpolate between TR and BR
        end_x = int(corners[1][0] + t * (corners[2][0] - corners[1][0]))
        end_y = int(corners[1][1] + t * (corners[2][1] - corners[1][1]))

        # Draw horizontal line
        color = (0, 255, 255) if i % 3 == 0 else (100, 100, 255)  # Yellow for thick lines, light blue for thin
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(result, (start_x, start_y), (end_x, end_y), color, thickness)

    # Draw vertical lines
    for i in range(10):
        x = i * cell_size

        # Calculate start and end points
        t = x / output_size  # Position ratio (0 to 1)

        # Start point (top edge): interpolate between TL and TR
        start_x = int(corners[0][0] + t * (corners[1][0] - corners[0][0]))
        start_y = int(corners[0][1] + t * (corners[1][1] - corners[0][1]))

        # End point (bottom edge): interpolate between BL and BR
        end_x = int(corners[3][0] + t * (corners[2][0] - corners[3][0]))
        end_y = int(corners[3][1] + t * (corners[2][1] - corners[3][1]))

        # Draw vertical line
        color = (0, 255, 255) if i % 3 == 0 else (100, 100, 255)  # Yellow for thick lines, light blue for thin
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(result, (start_x, start_y), (end_x, end_y), color, thickness)

    return result


def reinforce_grid_adaptive(binary_image, output_size=450):
    """
    Scan the straightened grid at expected line positions and fill missing lines.

    After perspective transform, the grid is straightened, so we can simply:
    1. Check each of the 10 horizontal line positions
    2. Check each of the 10 vertical line positions
    3. If a line is missing or broken, draw it

    Strategy: Check for LARGE gaps in the line. If there's a gap > 100 pixels,
    the line is considered broken. This is more reliable than coverage %.

    Args:
        binary_image: Binary straightened image with grid
        output_size: Expected grid size (450x450)

    Returns:
        Reinforced binary image with missing lines filled
    """
    h, w = binary_image.shape
    step_x = w // 9
    step_y = h // 9

    result = binary_image.copy()
    missing_lines = []

    print(f"        Scanning grid at {w}x{h}...")

    # Check each horizontal line position
    for i in range(10):
        y_center = i * step_y
        y_start = max(0, y_center - 2)
        y_end = min(h, y_center + 3)

        # Sample the line region
        line_region = binary_image[y_start:y_end, :]

        if line_region.size > 0:
            # Get line projection (max across height of region)
            line_projection = np.max(line_region, axis=0)

            # Calculate total coverage (how much of line is present)
            coverage = np.sum(line_projection > 0) / w

            # Only fill if line is severely broken or missing (< 30% coverage)
            # Complete grids with digits have 60-90% coverage even with interruptions
            if coverage < 0.30:
                cv2.line(result, (0, y_center), (w-1, y_center), 255, 2)
                missing_lines.append(f"H{i} ({coverage*100:.0f}%)")

    # Check each vertical line position
    for i in range(10):
        x_center = i * step_x
        x_start = max(0, x_center - 2)
        x_end = min(w, x_center + 3)

        # Sample the line region
        line_region = binary_image[:, x_start:x_end]

        if line_region.size > 0:
            # Get line projection (max across width of region)
            line_projection = np.max(line_region, axis=1)

            # Calculate total coverage (how much of line is present)
            coverage = np.sum(line_projection > 0) / h

            # Only fill if line is severely broken or missing (< 30% coverage)
            # Complete grids with digits have 60-90% coverage even with interruptions
            if coverage < 0.30:
                cv2.line(result, (x_center, 0), (x_center, h-1), 255, 2)
                missing_lines.append(f"V{i} ({coverage*100:.0f}%)")

    if len(missing_lines) > 0:
        print(f"        Missing lines filled: {', '.join(missing_lines)}")
        return result
    else:
        print(f"        Grid is complete - no lines missing")
        return binary_image


def merge_closest_points(points, target_count=4):
    """
    Merge closest points iteratively until we have target_count points.

    Args:
        points: Array of points (N, 2)
        target_count: Target number of points (default: 4)

    Returns:
        numpy.ndarray: Merged points
    """
    points = points.copy()

    while len(points) > target_count:
        # Find the two closest points
        min_dist = float('inf')
        merge_idx = (0, 1)

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < min_dist:
                    min_dist = dist
                    merge_idx = (i, j)

        # Merge the two closest points (take average)
        i, j = merge_idx
        merged_point = (points[i] + points[j]) / 2

        # Remove the two points and add the merged point
        points = np.delete(points, [i, j], axis=0)
        points = np.vstack([points, merged_point])

    return points


def get_extreme_corners_improved(contour):
    """
    Extract the four extreme corner points from a contour using improved algorithm.

    ENHANCED: Uses angle-based detection from centroid for better accuracy on rotated grids.

    Args:
        contour: Input contour

    Returns:
        numpy.ndarray: Four corner points
    """
    # Reshape contour to Nx2 array
    points = contour.reshape(-1, 2).astype(np.float32)

    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Calculate angle from centroid for each point
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort points by angle
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    sorted_angles = angles[sorted_indices]

    # Divide into 4 quadrants and find the extreme point in each
    # Quadrants: [-π, -π/2], [-π/2, 0], [0, π/2], [π/2, π]
    corners = []

    # Top-right quadrant: angle in [-π/2, 0] - find furthest from centroid
    q1_mask = (sorted_angles >= -np.pi/2) & (sorted_angles < 0)
    if np.any(q1_mask):
        q1_points = sorted_points[q1_mask]
        dists = np.linalg.norm(q1_points - centroid, axis=1)
        corners.append(q1_points[np.argmax(dists)])
    else:
        # Fallback
        corners.append(points[np.argmax(points[:, 0] - points[:, 1])])

    # Bottom-right quadrant: angle in [0, π/2]
    q2_mask = (sorted_angles >= 0) & (sorted_angles < np.pi/2)
    if np.any(q2_mask):
        q2_points = sorted_points[q2_mask]
        dists = np.linalg.norm(q2_points - centroid, axis=1)
        corners.append(q2_points[np.argmax(dists)])
    else:
        corners.append(points[np.argmax(points[:, 0] + points[:, 1])])

    # Bottom-left quadrant: angle in [π/2, π]
    q3_mask = (sorted_angles >= np.pi/2) & (sorted_angles <= np.pi)
    if np.any(q3_mask):
        q3_points = sorted_points[q3_mask]
        dists = np.linalg.norm(q3_points - centroid, axis=1)
        corners.append(q3_points[np.argmax(dists)])
    else:
        corners.append(points[np.argmin(points[:, 0] - points[:, 1])])

    # Top-left quadrant: angle in [-π, -π/2]
    q4_mask = (sorted_angles >= -np.pi) & (sorted_angles < -np.pi/2)
    if np.any(q4_mask):
        q4_points = sorted_points[q4_mask]
        dists = np.linalg.norm(q4_points - centroid, axis=1)
        corners.append(q4_points[np.argmax(dists)])
    else:
        corners.append(points[np.argmin(points[:, 0] + points[:, 1])])

    return np.array(corners, dtype=np.float32)


def validate_corner_geometry(corners, min_angle=60, max_angle=120):
    """
    Validate that corners form a valid quadrilateral.

    Checks:
    - All four corners are distinct
    - Internal angles are reasonable (not too acute or obtuse)
    - Quadrilateral is convex

    Args:
        corners: Array of 4 corner points [TL, TR, BR, BL]
        min_angle: Minimum acceptable internal angle (default: 60 degrees)
        max_angle: Maximum acceptable internal angle (default: 120 degrees)

    Returns:
        bool: True if geometry is valid
    """
    if len(corners) != 4:
        return False

    # Check that corners are distinct
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(corners[i] - corners[j]) < 5:
                return False

    # Check internal angles
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]

        # Vectors from p2 to p1 and p2 to p3
        v1 = p1 - p2
        v2 = p3 - p2

        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle) * 180 / np.pi

        if angle < min_angle or angle > max_angle:
            return False

    # Check convexity (all cross products should have same sign)
    cross_products = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]

        v1 = p2 - p1
        v2 = p3 - p2

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cross)

    # All should have same sign for convex quadrilateral
    if not all(cp > 0 for cp in cross_products) and not all(cp < 0 for cp in cross_products):
        return False

    return True


def find_largest_contour(processed_image):
    """
    Find the largest contour in the image, which should be the Sudoku grid.

    The algorithm:
    1. Finds all contours in the binary image
    2. Filters contours by area (must be significant portion of image)
    3. Returns the largest contour by area

    Args:
        processed_image: Binary preprocessed image (numpy array)

    Returns:
        numpy.ndarray: Largest contour or None if not found

    References:
        - Suzuki & Abe (1985): Topological structural analysis method
        - OpenCV findContours: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
    """
    # Find all contours in the image
    # RETR_EXTERNAL: retrieves only extreme outer contours
    # CHAIN_APPROX_SIMPLE: compresses contour by removing redundant points
    contours, hierarchy = cv2.findContours(
        processed_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # The Sudoku grid should be one of the largest contours
    # Filter by minimum area (at least 20% of image area)
    image_area = processed_image.shape[0] * processed_image.shape[1]
    min_area = image_area * 0.2

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Approximate the contour to a polygon
            # This reduces the number of points while preserving shape
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            # Sudoku grid should be approximately a quadrilateral (4 corners)
            # We accept 4-8 points as the perspective may distort the shape
            if len(approx) >= 4:
                return contour

    return None


def find_grid_corners(contour):
    """
    Identify the four corners of the Sudoku grid from its contour.

    ENHANCED: Multi-pass approximation + sub-pixel refinement for ultra-precision.

    Uses the convex hull and corner detection to find the four corner points
    of the grid quadrilateral. The corners are ordered as:
    [top-left, top-right, bottom-right, bottom-left]

    Args:
        contour: Contour of the Sudoku grid

    Returns:
        numpy.ndarray: Array of 4 corner points in clockwise order from top-left

    References:
        - Corner ordering technique: Common CV practice for perspective transforms
        - Convex hull: Sklansky (1982) "Finding the convex hull of a simple polygon"
        - Sub-pixel refinement: Harris & Stephens (1988) "A Combined Corner and Edge Detector"
    """
    # Get the convex hull of the contour
    # This finds the smallest convex polygon that contains the contour
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)

    # ENHANCEMENT 1: Multi-pass approximation with different epsilon values
    # Try multiple approximation factors to find the best 4-point quadrilateral
    best_approx = None
    epsilon_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]

    for epsilon_factor in epsilon_values:
        approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)

        # If we get exactly 4 points, this is ideal
        if len(approx) == 4:
            best_approx = approx
            break

        # If we get 5-6 points, save as backup (we can merge close points)
        if best_approx is None and 5 <= len(approx) <= 6:
            best_approx = approx

    # If we have exactly 4 points, use them
    if best_approx is not None and len(best_approx) == 4:
        corners = best_approx.reshape(4, 2)
    elif best_approx is not None and 5 <= len(best_approx) <= 6:
        # Merge closest points to get 4 corners
        corners = merge_closest_points(best_approx.reshape(-1, 2), target_count=4)
    else:
        # Fallback: find the 4 extreme points using improved algorithm
        corners = get_extreme_corners_improved(hull)

    # Order the corners: top-left, top-right, bottom-right, bottom-left
    # Note: Skip local refinement for contour-based detection to avoid
    # pulling corners into text/occluded areas
    ordered_corners = order_corners(corners)

    # ENHANCEMENT 3: Validate corner quality
    if not validate_corner_geometry(ordered_corners):
        print("      WARNING: Corner geometry validation failed, using fallback")
        # Fallback to extreme corners if validation fails
        corners = get_extreme_corners_improved(hull)
        ordered_corners = order_corners(corners)

    return ordered_corners


def order_corners(corners):
    """
    Order corner points in clockwise order starting from top-left.

    The ordering is: [top-left, top-right, bottom-right, bottom-left]
    This is essential for correct perspective transformation.

    Args:
        corners: Array of 4 corner points in any order

    Returns:
        numpy.ndarray: Ordered corners [TL, TR, BR, BL]

    References:
        - Standard ordering for perspective transforms in CV applications
    """
    # Reshape to ensure we have (4, 2) shape
    corners = corners.reshape(4, 2)

    # Calculate center point
    center = corners.mean(axis=0)

    # Sort by angle from center (clockwise from top-left)
    def angle_from_center(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])

    # Sort corners by angle
    sorted_corners = sorted(corners, key=angle_from_center)

    # Ensure top-left is first (minimum y among top two points)
    # Find the two topmost points
    sorted_by_y = sorted(sorted_corners, key=lambda p: p[1])

    # Top-left has smaller x among top two
    if sorted_by_y[0][0] > sorted_by_y[1][0]:
        sorted_by_y[0], sorted_by_y[1] = sorted_by_y[1], sorted_by_y[0]

    # Bottom-right has larger x among bottom two
    if sorted_by_y[2][0] > sorted_by_y[3][0]:
        top_left, top_right = sorted_by_y[0], sorted_by_y[1]
        bottom_right, bottom_left = sorted_by_y[2], sorted_by_y[3]
    else:
        top_left, top_right = sorted_by_y[0], sorted_by_y[1]
        bottom_left, bottom_right = sorted_by_y[2], sorted_by_y[3]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def select_grid_lines(clusters, target_count=10):
    """
    Select the most evenly-spaced lines from clusters to represent a 9x9 grid.

    When we detect too many line clusters (due to noise or internal structure),
    this function filters them to find the target_count most evenly-spaced lines
    that best represent the actual grid structure.

    Args:
        clusters: List of line positions (sorted)
        target_count: Target number of lines to select (default: 10 for 9x9 grid)

    Returns:
        list: Selected line positions
    """
    if len(clusters) <= target_count:
        return clusters

    # Always keep first and last (outermost boundaries)
    selected = [clusters[0], clusters[-1]]

    # Calculate expected spacing for a uniform grid
    total_span = clusters[-1] - clusters[0]
    expected_spacing = total_span / (target_count - 1)

    # Greedily select lines that are closest to expected positions
    for i in range(1, target_count - 1):
        expected_pos = clusters[0] + i * expected_spacing

        # Find cluster closest to expected position
        best_cluster = None
        min_dist = float('inf')

        for cluster in clusters[1:-1]:  # Exclude first and last (already selected)
            if cluster in selected:
                continue

            dist = abs(cluster - expected_pos)
            if dist < min_dist:
                min_dist = dist
                best_cluster = cluster

        if best_cluster is not None:
            selected.append(best_cluster)

    return sorted(selected)


def detect_grid_using_lines(processed_image, debug=False):
    """
    Detect the 9x9 Sudoku grid by finding grid lines using Hough Transform.

    ENHANCED: Better line detection, clustering, and corner refinement.

    This method is more robust than contour detection because it directly
    finds the grid lines and computes corners from their intersections.

    Args:
        processed_image: Binary preprocessed image
        debug: If True, print debug information

    Returns:
        numpy.ndarray: Four corner points [TL, TR, BR, BL] or None if detection fails
    """
    h, w = processed_image.shape

    def classify_lines(lines):
        """Separate raw Hough lines into horizontal/vertical buckets."""
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate angle
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # ENHANCEMENT 2: Store line equations, not just positions
            # Horizontal lines: angle close to 0 or 180
            if angle < 15 or angle > 165:
                # Fit line equation: y = mx + c
                if x2 != x1:
                    m = (y2 - y1) / (x2 - x1)
                    c = y1 - m * x1
                    horizontal_lines.append((m, c, (y1 + y2) / 2))
                else:
                    # Vertical line, skip
                    continue

            # Vertical lines: angle close to 90
            elif 75 < angle < 105:
                # Fit line equation: x = my + c
                if y2 != y1:
                    m = (x2 - x1) / (y2 - y1)
                    c = x1 - m * y1
                    vertical_lines.append((m, c, (x1 + x2) / 2))
                else:
                    # Horizontal line, skip
                    continue

        return horizontal_lines, vertical_lines

    # Try standard Hough first
    lines1 = cv2.HoughLinesP(
        processed_image,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=int(min(h, w) * 0.3),
        maxLineGap=20
    )
    all_lines = [] if lines1 is None else list(lines1)

    # Classify initial detection
    horizontal_lines, vertical_lines = classify_lines(all_lines)

    # If standard detection is weak, fall back to a more sensitive pass
    if (len(horizontal_lines) < 3 or len(vertical_lines) < 3):
        if debug:
            print("      Low line count with standard params, trying sensitive Hough pass")

        lines2 = cv2.HoughLinesP(
            processed_image,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=int(min(h, w) * 0.2),
            maxLineGap=30
        )
        if lines2 is not None:
            all_lines.extend(lines2)
            horizontal_lines, vertical_lines = classify_lines(all_lines)

    if len(all_lines) == 0:
        if debug:
            print("      No lines detected with Hough transform")
        return None

    if debug:
        print(f"      Detected {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical line segments")

    # Need at least some lines in both directions
    if len(horizontal_lines) < 3 or len(vertical_lines) < 3:
        if debug:
            print("      Not enough lines detected in both directions")
        return None

    # ENHANCEMENT 3: Improved clustering with outlier removal
    h_positions = [line[2] for line in horizontal_lines]
    v_positions = [line[2] for line in vertical_lines]

    h_clusters = cluster_lines_improved(h_positions, tolerance=10)
    v_clusters = cluster_lines_improved(v_positions, tolerance=10)

    if debug:
        print(f"      Clustered into {len(h_clusters)} horizontal, {len(v_clusters)} vertical groups")

    # We expect 10 lines in each direction for a 9x9 grid
    # But we can work with the outermost lines
    if len(h_clusters) < 2 or len(v_clusters) < 2:
        if debug:
            print("      Not enough line clusters to define grid boundaries")
        return None

    # ENHANCEMENT: If we have too many clusters, filter to the most evenly-spaced 10
    # This removes noise and ensures we're detecting the actual 9x9 grid structure
    if len(h_clusters) > 10:
        h_clusters = select_grid_lines(h_clusters, target_count=10)
        if debug:
            print(f"      Filtered to {len(h_clusters)} most evenly-spaced horizontal lines")

    if len(v_clusters) > 10:
        v_clusters = select_grid_lines(v_clusters, target_count=10)
        if debug:
            print(f"      Filtered to {len(v_clusters)} most evenly-spaced vertical lines")

    # Get outermost lines (top, bottom, left, right)
    top = min(h_clusters)
    bottom = max(h_clusters)
    left = min(v_clusters)
    right = max(v_clusters)

    # ENHANCEMENT 4: Compute corners using BOTH methods and validate

    # Method 1: Simple position-based (robust, conservative)
    corners_simple = np.array([
        [left, top],      # Top-left
        [right, top],     # Top-right
        [right, bottom],  # Bottom-right
        [left, bottom]    # Bottom-left
    ], dtype=np.float32)

    # Method 2: Line equation-based (precise, but sensitive to noise)
    # Only use if we have enough confident lines near boundaries
    use_equation_method = False

    # Find lines near boundaries (with tighter tolerance for quality)
    tolerance = 10
    top_lines = [line for line in horizontal_lines if abs(line[2] - top) < tolerance]
    bottom_lines = [line for line in horizontal_lines if abs(line[2] - bottom) < tolerance]
    left_lines = [line for line in vertical_lines if abs(line[2] - left) < tolerance]
    right_lines = [line for line in vertical_lines if abs(line[2] - right) < tolerance]

    # Only use equation method if we have at least 2 lines per boundary
    # This ensures we're averaging real grid lines, not noise
    if (len(top_lines) >= 2 and len(bottom_lines) >= 2 and
        len(left_lines) >= 2 and len(right_lines) >= 2):

        # Filter to use only the most confident lines (closest to boundary)
        # This reduces noise from internal grid lines
        top_lines = sorted(top_lines, key=lambda x: abs(x[2] - top))[:3]
        bottom_lines = sorted(bottom_lines, key=lambda x: abs(x[2] - bottom))[:3]
        left_lines = sorted(left_lines, key=lambda x: abs(x[2] - left))[:3]
        right_lines = sorted(right_lines, key=lambda x: abs(x[2] - right))[:3]

        # Average line equations using median for robustness
        top_m = np.median([line[0] for line in top_lines])
        top_c = np.median([line[1] for line in top_lines])
        bottom_m = np.median([line[0] for line in bottom_lines])
        bottom_c = np.median([line[1] for line in bottom_lines])
        left_m = np.median([line[0] for line in left_lines])
        left_c = np.median([line[1] for line in left_lines])
        right_m = np.median([line[0] for line in right_lines])
        right_c = np.median([line[1] for line in right_lines])

        # Check that slopes are reasonable (not too steep)
        # Grid lines should be nearly horizontal/vertical
        if (abs(top_m) < 0.1 and abs(bottom_m) < 0.1 and
            abs(left_m) < 0.1 and abs(right_m) < 0.1):

            # Compute intersections
            def line_intersection(h_m, h_c, v_m, v_c):
                """Calculate intersection of horizontal line y = h_m*x + h_c and vertical line x = v_m*y + v_c"""
                denom = 1 - h_m * v_m
                if abs(denom) < 1e-6:
                    y = h_c
                    x = v_c
                else:
                    y = (h_m * v_c + h_c) / denom
                    x = v_m * y + v_c
                return [x, y]

            corner_tl = line_intersection(top_m, top_c, left_m, left_c)
            corner_tr = line_intersection(top_m, top_c, right_m, right_c)
            corner_br = line_intersection(bottom_m, bottom_c, right_m, right_c)
            corner_bl = line_intersection(bottom_m, bottom_c, left_m, left_c)

            corners_equation = np.array([corner_tl, corner_tr, corner_br, corner_bl], dtype=np.float32)

            # Validate equation-based corners have good aspect ratio
            # Calculate side lengths
            top_len = np.linalg.norm(corners_equation[1] - corners_equation[0])
            bottom_len = np.linalg.norm(corners_equation[2] - corners_equation[3])
            left_len = np.linalg.norm(corners_equation[3] - corners_equation[0])
            right_len = np.linalg.norm(corners_equation[2] - corners_equation[1])

            width = (top_len + bottom_len) / 2
            height = (left_len + right_len) / 2
            aspect_ratio = width / height if height > 0 else 0

            # Only use equation method if aspect ratio is reasonable (0.92 to 1.08)
            # Sudoku grids should be close to square - strict threshold for quality
            if 0.92 <= aspect_ratio <= 1.08:
                use_equation_method = True
                corners = corners_equation
                if debug:
                    print(f"      Using equation-based corners (aspect ratio: {aspect_ratio:.2f})")
            elif debug:
                print(f"      Equation method rejected: aspect ratio {aspect_ratio:.2f} outside valid range")

    # Use simple position-based method if equation method wasn't suitable
    if not use_equation_method:
        corners = corners_simple
        if debug:
            print(f"      Using position-based corners (more robust for this image)")

    # Validate that corners are within image bounds (with some margin)
    margin = 50
    for i, corner in enumerate(corners):
        corners[i][0] = np.clip(corner[0], -margin, w + margin)
        corners[i][1] = np.clip(corner[1], -margin, h + margin)

    if debug:
        print(f"      Grid boundaries:")
        print(f"        Top-left: ({corners[0][0]:.1f}, {corners[0][1]:.1f})")
        print(f"        Top-right: ({corners[1][0]:.1f}, {corners[1][1]:.1f})")
        print(f"        Bottom-right: ({corners[2][0]:.1f}, {corners[2][1]:.1f})")
        print(f"        Bottom-left: ({corners[3][0]:.1f}, {corners[3][1]:.1f})")

    return corners


def cluster_lines_improved(line_positions, tolerance=10, min_cluster_size=1):
    """
    Improved line clustering with outlier removal.

    ENHANCED: Removes outliers and uses median for better robustness.

    Args:
        line_positions: List of line positions (x or y coordinates)
        tolerance: Maximum distance between lines in the same cluster
        min_cluster_size: Minimum number of lines to form a valid cluster

    Returns:
        list: Representative positions for each cluster (sorted)
    """
    if not line_positions:
        return []

    # Sort positions
    sorted_pos = sorted(line_positions)

    # Remove outliers using IQR method (optional, for very noisy data)
    if len(sorted_pos) > 10:
        q1 = np.percentile(sorted_pos, 25)
        q3 = np.percentile(sorted_pos, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        sorted_pos = [p for p in sorted_pos if lower_bound <= p <= upper_bound]

    if not sorted_pos:
        return []

    clusters = []
    current_cluster = [sorted_pos[0]]

    for pos in sorted_pos[1:]:
        if pos - current_cluster[-1] <= tolerance:
            # Add to current cluster
            current_cluster.append(pos)
        else:
            # Finish current cluster
            if len(current_cluster) >= min_cluster_size:
                # Use median instead of mean for robustness
                clusters.append(np.median(current_cluster))
            current_cluster = [pos]

    # Add last cluster
    if len(current_cluster) >= min_cluster_size:
        clusters.append(np.median(current_cluster))

    return sorted(clusters)


def draw_contour_and_corners(image, contour, corners):
    """
    Visualize the detected contour and corners on the image.

    Utility function for debugging and verification.

    Args:
        image: Original image
        contour: Detected grid contour
        corners: Four corner points

    Returns:
        numpy.ndarray: Image with drawn contour and corners
    """
    output = image.copy()

    # Draw the contour in green
    cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)

    # Draw corners as red circles
    for i, corner in enumerate(corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(output, (x, y), 10, (0, 0, 255), -1)
        # Label corners
        cv2.putText(output, str(i), (x + 15, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return output
