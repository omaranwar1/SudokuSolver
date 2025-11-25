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


def validate_grid_is_square(corners, tolerance=0.2):
    """
    Validate that the detected grid is approximately square.

    Args:
        corners: Array of 4 corner points [TL, TR, BR, BL]
        tolerance: How much deviation from square is acceptable (0.2 = 20%)

    Returns:
        tuple: (is_valid, aspect_ratio, message)
    """
    
    top = np.linalg.norm(corners[1] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])

    width = (top + bottom) / 2
    height = (left + right) / 2


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

        t = y / output_size  

        
        start_x = int(corners[0][0] + t * (corners[3][0] - corners[0][0]))
        start_y = int(corners[0][1] + t * (corners[3][1] - corners[0][1]))

        
        end_x = int(corners[1][0] + t * (corners[2][0] - corners[1][0]))
        end_y = int(corners[1][1] + t * (corners[2][1] - corners[1][1]))

        color = (0, 255, 255) if i % 3 == 0 else (100, 100, 255)  # Yellow for thick lines, light blue for thin
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(result, (start_x, start_y), (end_x, end_y), color, thickness)

   
    for i in range(10):
        x = i * cell_size

       
        t = x / output_size  

      
        start_x = int(corners[0][0] + t * (corners[1][0] - corners[0][0]))
        start_y = int(corners[0][1] + t * (corners[1][1] - corners[0][1]))

        
        end_x = int(corners[3][0] + t * (corners[2][0] - corners[3][0]))
        end_y = int(corners[3][1] + t * (corners[2][1] - corners[3][1]))

        
        color = (0, 255, 255) if i % 3 == 0 else (100, 100, 255)  # Yellow for thick lines, light blue for thin
        thickness = 2 if i % 3 == 0 else 1
        cv2.line(result, (start_x, start_y), (end_x, end_y), color, thickness)

    return result


def draw_template_grid(width=450, height=450, line_thickness=2, bold_thickness=3):
    """
    Create a clean 9x9 grid mask sized to the provided width/height.
    """
    grid = np.zeros((height, width), dtype=np.uint8)
    step_x = width / 9.0
    step_y = height / 9.0

    for i in range(10):
        thickness = bold_thickness if i % 3 == 0 else line_thickness
        y = int(round(i * step_y))
        y = min(y, height - 1)
        cv2.line(grid, (0, y), (width - 1, y), 255, thickness)

        x = int(round(i * step_x))
        x = min(x, width - 1)
        cv2.line(grid, (x, 0), (x, height - 1), 255, thickness)

    return grid


def reinforce_grid_morphological(binary_image, output_size=450):
    """
    Reinforce grid lines - extends existing lines AND fills missing ones.

    This method:
    1. Detects and extends existing lines
    2. Fills in completely missing lines at expected positions
    3. Preserves digits

    Args:
        binary_image: Binary image with grid
        output_size: Expected grid size

    Returns:
        Reinforced binary image
    """
    h, w = binary_image.shape
    step_x = w // 9
    step_y = h // 9

    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 9, 1))
    detected_horizontal = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 9))
    detected_vertical = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

   
    horizontal_extend = cv2.dilate(detected_horizontal, np.ones((1, 15), np.uint8), iterations=1)
    vertical_extend = cv2.dilate(detected_vertical, np.ones((15, 1), np.uint8), iterations=1)

   
    grid_mask = np.zeros_like(binary_image)

    
    total_extended_pixels = np.sum(horizontal_extend > 0) + np.sum(vertical_extend > 0)

    if total_extended_pixels < 1000:
       
        print(f"        Grid completely missing - drawing full template")
        grid_mask = draw_template_grid(width=w, height=h, line_thickness=2, bold_thickness=4)
    else:
       
        print(f"        Grid partially present - filling missing lines only")

        
        for i in range(10):
            y_pos = i * step_y
            
            y_start = max(0, y_pos - 2)
            y_end = min(h, y_pos + 3)
            line_region = horizontal_extend[y_start:y_end, :]

          
            if line_region.size > 0:
                coverage = np.sum(line_region > 0) / (line_region.shape[0] * w)
                if coverage < 0.10:
                   
                    cv2.line(grid_mask, (0, y_pos), (w-1, y_pos), 255, 2)


        for i in range(10):
            x_pos = i * step_x
            
            x_start = max(0, x_pos - 2)
            x_end = min(w, x_pos + 3)
            line_region = vertical_extend[:, x_start:x_end]

           
            if line_region.size > 0:
                coverage = np.sum(line_region > 0) / (line_region.shape[1] * h)
                if coverage < 0.10:
                    
                    cv2.line(grid_mask, (x_pos, 0), (x_pos, h-1), 255, 2)

   
    combined_grid = cv2.bitwise_or(horizontal_extend, vertical_extend)
    combined_grid = cv2.bitwise_or(combined_grid, grid_mask)

   
    grid_buffer = cv2.dilate(combined_grid, np.ones((3, 3), np.uint8), iterations=1)
    digits_only = cv2.bitwise_and(binary_image, cv2.bitwise_not(grid_buffer))

    
    result = cv2.bitwise_or(digits_only, combined_grid)

    return result


def reinforce_grid_hough(binary_image, output_size=450):
    """
    Reinforce grid using Hough line detection and extension.

    This method detects existing lines, extends them, and fills gaps.
    More intelligent than template overlay.

    Args:
        binary_image: Binary image with grid
        output_size: Expected grid size

    Returns:
        Reinforced binary image
    """
    h, w = binary_image.shape

    
    edges = cv2.Canny(binary_image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=min(h, w) // 5, maxLineGap=20)

    if lines is None:
        return binary_image

    
    result = binary_image.copy()

    
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        if angle < 15 or angle > 165:  # Horizontal
            horizontal_lines.append((x1, y1, x2, y2))
        elif 75 < angle < 105:  # Vertical
            vertical_lines.append((x1, y1, x2, y2))

    
    line_mask = np.zeros_like(binary_image)
    for x1, y1, x2, y2 in horizontal_lines:
        y_avg = (y1 + y2) // 2
        cv2.line(line_mask, (0, y_avg), (w-1, y_avg), 255, 2)

    
    for x1, y1, x2, y2 in vertical_lines:
        x_avg = (x1 + x2) // 2
        cv2.line(line_mask, (x_avg, 0), (x_avg, h-1), 255, 2)

   
    result = cv2.bitwise_or(binary_image, line_mask)

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

   
    for i in range(10):
        y_center = i * step_y
        y_start = max(0, y_center - 2)
        y_end = min(h, y_center + 3)

        
        line_region = binary_image[y_start:y_end, :]

        if line_region.size > 0:
            
            line_projection = np.max(line_region, axis=0)

            
            coverage = np.sum(line_projection > 0) / w

            
            if coverage < 0.30:
                cv2.line(result, (0, y_center), (w-1, y_center), 255, 2)
                missing_lines.append(f"H{i} ({coverage*100:.0f}%)")

    
    for i in range(10):
        x_center = i * step_x
        x_start = max(0, x_center - 2)
        x_end = min(w, x_center + 3)

       
        line_region = binary_image[:, x_start:x_end]

        if line_region.size > 0:
           
            line_projection = np.max(line_region, axis=1)

            
            coverage = np.sum(line_projection > 0) / h

            
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


        i, j = merge_idx
        merged_point = (points[i] + points[j]) / 2

        
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
   
    points = contour.reshape(-1, 2).astype(np.float32)

    
    centroid = np.mean(points, axis=0)

    
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    sorted_angles = angles[sorted_indices]

    
    corners = []

    
    q1_mask = (sorted_angles >= -np.pi/2) & (sorted_angles < 0)
    if np.any(q1_mask):
        q1_points = sorted_points[q1_mask]
        dists = np.linalg.norm(q1_points - centroid, axis=1)
        corners.append(q1_points[np.argmax(dists)])
    else:
        
        corners.append(points[np.argmax(points[:, 0] - points[:, 1])])

   
    q2_mask = (sorted_angles >= 0) & (sorted_angles < np.pi/2)
    if np.any(q2_mask):
        q2_points = sorted_points[q2_mask]
        dists = np.linalg.norm(q2_points - centroid, axis=1)
        corners.append(q2_points[np.argmax(dists)])
    else:
        corners.append(points[np.argmax(points[:, 0] + points[:, 1])])

    
    q3_mask = (sorted_angles >= np.pi/2) & (sorted_angles <= np.pi)
    if np.any(q3_mask):
        q3_points = sorted_points[q3_mask]
        dists = np.linalg.norm(q3_points - centroid, axis=1)
        corners.append(q3_points[np.argmax(dists)])
    else:
        corners.append(points[np.argmin(points[:, 0] - points[:, 1])])

    
    q4_mask = (sorted_angles >= -np.pi) & (sorted_angles < -np.pi/2)
    if np.any(q4_mask):
        q4_points = sorted_points[q4_mask]
        dists = np.linalg.norm(q4_points - centroid, axis=1)
        corners.append(q4_points[np.argmax(dists)])
    else:
        corners.append(points[np.argmin(points[:, 0] + points[:, 1])])

    return np.array(corners, dtype=np.float32)


def refine_corners_locally(corners, contour, search_radius=15):
    """
    Refine corner positions by finding the true corner within a local region.

    Uses Harris corner detection in a local window around each initial corner.

    Args:
        corners: Initial corner positions (4, 2)
        contour: Original contour
        search_radius: Radius to search around each corner (default: 15)

    Returns:
        numpy.ndarray: Refined corner positions
    """
    
    points = contour.reshape(-1, 2)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    
    padding = search_radius + 10
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = int(x_max) + padding
    y_max = int(y_max) + padding

   
    mask_height = y_max - y_min
    mask_width = x_max - x_min
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

   
    adjusted_contour = contour.copy()
    adjusted_contour[:, :, 0] -= x_min
    adjusted_contour[:, :, 1] -= y_min
    cv2.drawContours(mask, [adjusted_contour], -1, 255, 2)

    refined_corners = []

    for corner in corners:
        cx, cy = int(corner[0]), int(corner[1])

       
        cx_mask = cx - x_min
        cy_mask = cy - y_min

        
        x1 = max(0, cx_mask - search_radius)
        y1 = max(0, cy_mask - search_radius)
        x2 = min(mask_width, cx_mask + search_radius)
        y2 = min(mask_height, cy_mask + search_radius)

        
        window = mask[y1:y2, x1:x2]

        if window.size == 0:
            refined_corners.append(corner)
            continue

        
        window_float = np.float32(window)
        harris = cv2.cornerHarris(window_float, blockSize=3, ksize=3, k=0.04)

        
        if harris.max() > 0:
            max_loc = np.unravel_index(harris.argmax(), harris.shape)
            
            refined_y = max_loc[0] + y1 + y_min
            refined_x = max_loc[1] + x1 + x_min
            refined_corners.append([refined_x, refined_y])
        else:
            
            refined_corners.append(corner)

    return np.array(refined_corners, dtype=np.float32)


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

   
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(corners[i] - corners[j]) < 5:
                return False

    
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        p3 = corners[(i + 2) % 4]

       
        v1 = p1 - p2
        v2 = p3 - p2

        
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
   
    contours, hierarchy = cv2.findContours(
        processed_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

   
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    
    image_area = processed_image.shape[0] * processed_image.shape[1]
    min_area = image_area * 0.2

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            
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
    
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)

    
    best_approx = None
    epsilon_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05]

    for epsilon_factor in epsilon_values:
        approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)

        
        if len(approx) == 4:
            best_approx = approx
            break

        
        if best_approx is None and 5 <= len(approx) <= 6:
            best_approx = approx

   
    if best_approx is not None and len(best_approx) == 4:
        corners = best_approx.reshape(4, 2)
    elif best_approx is not None and 5 <= len(best_approx) <= 6:
       
        corners = merge_closest_points(best_approx.reshape(-1, 2), target_count=4)
    else:
       
        corners = get_extreme_corners_improved(hull)

    
    ordered_corners = order_corners(corners)

   
    if not validate_corner_geometry(ordered_corners):
        print("      WARNING: Corner geometry validation failed, using fallback")
        # Fallback to extreme corners if validation fails
        corners = get_extreme_corners_improved(hull)
        ordered_corners = order_corners(corners)

    return ordered_corners


def get_extreme_corners(contour):
    """
    Extract the four extreme corner points from a contour.

    Finds the points with minimum/maximum x and y coordinates to identify corners.

    Args:
        contour: Input contour

    Returns:
        numpy.ndarray: Four corner points
    """
    
    points = contour.reshape(-1, 2)

    
    top_left = points[np.argmin(points[:, 0] + points[:, 1])]  # Min sum of x+y
    top_right = points[np.argmin(points[:, 1] - points[:, 0])]  # Min y-x
    bottom_left = points[np.argmin(points[:, 0] - points[:, 1])]  # Min x-y
    bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]  # Max sum of x+y

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


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
   
    corners = corners.reshape(4, 2)

    
    center = corners.mean(axis=0)

    # Sort by angle from center (clockwise from top-left)
    def angle_from_center(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])

    # Sort corners by angle
    sorted_corners = sorted(corners, key=angle_from_center)

    
    sorted_by_y = sorted(sorted_corners, key=lambda p: p[1])

    
    if sorted_by_y[0][0] > sorted_by_y[1][0]:
        sorted_by_y[0], sorted_by_y[1] = sorted_by_y[1], sorted_by_y[0]

   
    if sorted_by_y[2][0] > sorted_by_y[3][0]:
        top_left, top_right = sorted_by_y[0], sorted_by_y[1]
        bottom_right, bottom_left = sorted_by_y[2], sorted_by_y[3]
    else:
        top_left, top_right = sorted_by_y[0], sorted_by_y[1]
        bottom_left, bottom_right = sorted_by_y[2], sorted_by_y[3]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def remove_line_outliers(clusters, image_dimension=None):
    """
    Remove outlier lines that don't fit the pattern of a 9x9 grid.

    A 9x9 Sudoku grid should have 10 evenly-spaced lines. Find the largest
    group of evenly-spaced lines and remove everything else.

    Args:
        clusters: List of line positions (sorted)
        image_dimension: Image dimension in this direction (for span validation)

    Returns:
        list: Filtered line positions with outliers removed
    """
    if len(clusters) <= 10:
        return clusters

    clusters = sorted(clusters)

   
    best_group = clusters
    best_score = float('inf')
    min_span_required = image_dimension * 0.65 if image_dimension else 0  

    
    for start_idx in range(len(clusters) - 9):  
        
        group = [clusters[start_idx]]
        gaps = []

        
        for i in range(start_idx + 1, len(clusters)):
            gap = clusters[i] - group[-1]

            if len(gaps) == 0:
                # First gap - just add it
                group.append(clusters[i])
                gaps.append(gap)
            else:
                
                median_gap = np.median(gaps)
                if 0.5 * median_gap <= gap <= 1.5 * median_gap:
                    group.append(clusters[i])
                    gaps.append(gap)
                elif gap > 1.5 * median_gap:
                    
                    break


        if len(group) >= 10:
            span = group[-1] - group[0]

            
            if image_dimension and span < min_span_required:
                continue

            gap_std = np.std(gaps) if len(gaps) > 1 else 0
           
            score = gap_std + abs(len(group) - 10) * 10

            
            if image_dimension:
                span_penalty = max(0, min_span_required - span) / 10
                score += span_penalty

            if score < best_score:
                best_score = score
                best_group = group[:10] if len(group) > 10 else group

    return sorted(best_group)


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

    
    selected = [clusters[0], clusters[-1]]

    
    total_span = clusters[-1] - clusters[0]
    expected_spacing = total_span / (target_count - 1)

   
    for i in range(1, target_count - 1):
        expected_pos = clusters[0] + i * expected_spacing

        
        best_cluster = None
        min_dist = float('inf')

        for cluster in clusters[1:-1]:  
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

            
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

           
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
                
                if y2 != y1:
                    m = (x2 - x1) / (y2 - y1)
                    c = x1 - m * y1
                    vertical_lines.append((m, c, (x1 + x2) / 2))
                else:
                    # Horizontal line, skip
                    continue

        return horizontal_lines, vertical_lines

   
    lines1 = cv2.HoughLinesP(
        processed_image,
        rho=1,
        theta=np.pi/180,
        threshold=100,
        minLineLength=int(min(h, w) * 0.3),
        maxLineGap=20
    )
    all_lines = [] if lines1 is None else list(lines1)

    
    horizontal_lines, vertical_lines = classify_lines(all_lines)

    
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

    
    if len(horizontal_lines) < 3 or len(vertical_lines) < 3:
        if debug:
            print("      Not enough lines detected in both directions")
        return None

    
    h_positions = [line[2] for line in horizontal_lines]
    v_positions = [line[2] for line in vertical_lines]

    h_clusters = cluster_lines_improved(h_positions, tolerance=10)
    v_clusters = cluster_lines_improved(v_positions, tolerance=10)

    if debug:
        print(f"      Clustered into {len(h_clusters)} horizontal, {len(v_clusters)} vertical groups")

    
    if len(h_clusters) < 2 or len(v_clusters) < 2:
        if debug:
            print("      Not enough line clusters to define grid boundaries")
        return None

   
    if len(h_clusters) > 10:
        h_clusters = select_grid_lines(h_clusters, target_count=10)
        if debug:
            print(f"      Filtered to {len(h_clusters)} most evenly-spaced horizontal lines")

    if len(v_clusters) > 10:
        v_clusters = select_grid_lines(v_clusters, target_count=10)
        if debug:
            print(f"      Filtered to {len(v_clusters)} most evenly-spaced vertical lines")

    
    top = min(h_clusters)
    bottom = max(h_clusters)
    left = min(v_clusters)
    right = max(v_clusters)

   
    corners_simple = np.array([
        [left, top],      # Top-left
        [right, top],     # Top-right
        [right, bottom],  # Bottom-right
        [left, bottom]    # Bottom-left
    ], dtype=np.float32)

   
    use_equation_method = False

   
    tolerance = 10
    top_lines = [line for line in horizontal_lines if abs(line[2] - top) < tolerance]
    bottom_lines = [line for line in horizontal_lines if abs(line[2] - bottom) < tolerance]
    left_lines = [line for line in vertical_lines if abs(line[2] - left) < tolerance]
    right_lines = [line for line in vertical_lines if abs(line[2] - right) < tolerance]

    
    if (len(top_lines) >= 2 and len(bottom_lines) >= 2 and
        len(left_lines) >= 2 and len(right_lines) >= 2):

       
        top_lines = sorted(top_lines, key=lambda x: abs(x[2] - top))[:3]
        bottom_lines = sorted(bottom_lines, key=lambda x: abs(x[2] - bottom))[:3]
        left_lines = sorted(left_lines, key=lambda x: abs(x[2] - left))[:3]
        right_lines = sorted(right_lines, key=lambda x: abs(x[2] - right))[:3]

       
        top_m = np.median([line[0] for line in top_lines])
        top_c = np.median([line[1] for line in top_lines])
        bottom_m = np.median([line[0] for line in bottom_lines])
        bottom_c = np.median([line[1] for line in bottom_lines])
        left_m = np.median([line[0] for line in left_lines])
        left_c = np.median([line[1] for line in left_lines])
        right_m = np.median([line[0] for line in right_lines])
        right_c = np.median([line[1] for line in right_lines])


        if (abs(top_m) < 0.1 and abs(bottom_m) < 0.1 and
            abs(left_m) < 0.1 and abs(right_m) < 0.1):

            
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

            
            top_len = np.linalg.norm(corners_equation[1] - corners_equation[0])
            bottom_len = np.linalg.norm(corners_equation[2] - corners_equation[3])
            left_len = np.linalg.norm(corners_equation[3] - corners_equation[0])
            right_len = np.linalg.norm(corners_equation[2] - corners_equation[1])

            width = (top_len + bottom_len) / 2
            height = (left_len + right_len) / 2
            aspect_ratio = width / height if height > 0 else 0

           
            if 0.92 <= aspect_ratio <= 1.08:
                use_equation_method = True
                corners = corners_equation
                if debug:
                    print(f"      Using equation-based corners (aspect ratio: {aspect_ratio:.2f})")
            elif debug:
                print(f"      Equation method rejected: aspect ratio {aspect_ratio:.2f} outside valid range")

   
    if not use_equation_method:
        corners = corners_simple
        if debug:
            print(f"      Using position-based corners (more robust for this image)")

    
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


def cluster_lines(line_positions, tolerance=10):
    """
    Cluster line positions that are close together.

    Args:
        line_positions: List of line positions (x or y coordinates)
        tolerance: Maximum distance between lines in the same cluster

    Returns:
        list: Representative positions for each cluster (sorted)
    """
    if not line_positions:
        return []

    # Sort positions
    sorted_pos = sorted(line_positions)

    clusters = []
    current_cluster = [sorted_pos[0]]

    for pos in sorted_pos[1:]:
        if pos - current_cluster[-1] <= tolerance:
            # Add to current cluster
            current_cluster.append(pos)
        else:
            # Start new cluster
            clusters.append(np.mean(current_cluster))
            current_cluster = [pos]

    # Add last cluster
    clusters.append(np.mean(current_cluster))

    return sorted(clusters)


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

    
    sorted_pos = sorted(line_positions)

    
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
          
            current_cluster.append(pos)
        else:
            
            if len(current_cluster) >= min_cluster_size:
                
                clusters.append(np.median(current_cluster))
            current_cluster = [pos]

   
    if len(current_cluster) >= min_cluster_size:
        clusters.append(np.median(current_cluster))

    return sorted(clusters)


def detect_grid_with_hough(processed_image):
    """
    Alternative method: Detect grid using Hough Line Transform.

    The Hough transform detects straight lines in the image, which can be used
    to find the grid lines and infer the outer frame.

    Args:
        processed_image: Binary preprocessed image

    Returns:
        list: Detected lines in the format [rho, theta]

    References:
        - Hough, P.V.C. (1962): Original Hough transform patent
        - Duda & Hart (1972): "Use of the Hough transformation to detect lines"
        - OpenCV HoughLines: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
    """
   
    lines = cv2.HoughLines(processed_image, 1, np.pi / 180, 200)

    return lines


def detect_blocked_corners(corners_contour, corners_lines, processed_image, debug=False):
    """
    Detect if corners are blocked/occluded (e.g., by a coin or other object).

    This function checks for corner displacement between contour and line-based methods.
    Significant displacement indicates occlusion affecting contour detection.

    Args:
        corners_contour: Corners detected by contour method
        corners_lines: Corners detected by line-based method
        processed_image: Binary processed image
        debug: If True, print debug information

    Returns:
        tuple: (is_blocked, reason_string)
    """
    if corners_contour is None or corners_lines is None:
        return False, "Cannot compare - one method failed"

    # Calculate corner displacement between methods
    # If corners are significantly displaced, it suggests occlusion affecting contour detection
    max_displacement = 0
    for i in range(4):
        displacement = np.linalg.norm(corners_contour[i] - corners_lines[i])
        max_displacement = max(max_displacement, displacement)

    # Calculate image diagonal for relative threshold
    h, w = processed_image.shape
    img_diagonal = np.sqrt(h**2 + w**2)
    displacement_ratio = max_displacement / img_diagonal

    if debug:
        print(f"      [Blocked Corner Detection]")
        print(f"        Max corner displacement: {max_displacement:.1f} pixels ({displacement_ratio*100:.1f}% of diagonal)")

    
        print(f"        ✓ Occlusion detected: corners displaced by {displacement_ratio*100:.1f}% of diagonal")
        return True, f"Corner blocked: corners displaced by {displacement_ratio*100:.1f}% of image diagonal"

    if debug:
        print(f"        ✗ No corner blockage detected")

    return False, "No corner blockage detected"


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
        
        cv2.putText(output, str(i), (x + 15, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return output
    
