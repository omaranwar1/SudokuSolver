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
from typing import Dict, List, Tuple, Optional


def build_digit_templates(size: int = 28, 
                          thicknesses: Tuple[int, ...] = (1, 2, 3),
                          fonts: Optional[Tuple[int, ...]] = None,
                          scales: Tuple[float, ...] = (0.9, 1.0, 1.1, 1.2),
                          rotations: Tuple[float, ...] = (0,)) -> Dict[int, List[np.ndarray]]:
    """
    Generate diverse digit templates (1-9) with multiple variations.
    
    Args:
        size: Template image size (size x size)
        thicknesses: Line thickness values to try
        fonts: OpenCV font styles (None = use multiple defaults)
        scales: Font scale factors
        rotations: Rotation angles in degrees (default: no rotation)
        
    Returns:
        Dictionary mapping digit -> list of template variants
    """
    if fonts is None:
        # Reduced to 2 most reliable fonts to minimize false positives
        fonts = (
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX
        )
    
    templates: Dict[int, List[np.ndarray]] = {}
    
    for digit in range(1, 10):
        variants = []
        
        for font in fonts:
            for thickness in thicknesses:
                for scale in scales:
                    for rotation in rotations:
                        # Create base canvas
                        canvas = np.zeros((size, size), dtype=np.uint8)
                        
                        # Render digit centered
                        text = str(digit)
                        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                        x = (size - text_size[0]) // 2
                        y = (size + text_size[1]) // 2
                        
                        cv2.putText(
                            canvas,
                            text,
                            (x, y),
                            font,
                            scale,
                            255,
                            thickness,
                            cv2.LINE_AA
                        )
                        
                        # Apply slight rotation if needed
                        if abs(rotation) > 0.1:
                            center = (size // 2, size // 2)
                            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
                            canvas = cv2.warpAffine(canvas, M, (size, size))
                        
                        # Normalize template
                        if canvas.max() > 0:
                            canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
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


def count_holes(binary_image: np.ndarray) -> int:
    """
    Count the number of holes in a binary image using Euler number.
    
    Holes are enclosed regions (like the hole in 0, 6, 8, 9).
    """
    # Find contours with hierarchy to detect holes
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None or len(contours) == 0:
        return 0
    
    # Count contours that have a parent (i.e., they're holes)
    holes = 0
    for i, h in enumerate(hierarchy[0]):
        # h[3] is the parent index; if >= 0, this contour is inside another
        if h[3] >= 0:
            holes += 1
    
    return holes


def compute_symmetry_score(binary_image: np.ndarray, axis: str = 'vertical') -> float:
    """
    Compute symmetry score along vertical or horizontal axis.
    
    Returns value between 0 (not symmetric) and 1 (perfectly symmetric).
    """
    if binary_image.size == 0:
        return 0.0
    
    if axis == 'vertical':
        # Flip horizontally
        flipped = cv2.flip(binary_image, 1)
    else:  # horizontal
        # Flip vertically
        flipped = cv2.flip(binary_image, 0)
    
    # Compute overlap
    intersection = cv2.bitwise_and(binary_image, flipped)
    union = cv2.bitwise_or(binary_image, flipped)
    
    union_pixels = cv2.countNonZero(union)
    if union_pixels == 0:
        return 0.0
    
    intersection_pixels = cv2.countNonZero(intersection)
    return intersection_pixels / union_pixels


def analyze_digit_topology(binary_image: np.ndarray) -> Dict[str, float]:
    """
    Extract topological and geometric features from a digit image.
    
    Returns dictionary of features useful for digit classification.
    """
    features = {}
    
    # 1. Hole count (Euler number method)
    features['holes'] = count_holes(binary_image)
    
    # 2. Aspect ratio
    h, w = binary_image.shape
    if w > 0:
        features['aspect_ratio'] = h / w
    else:
        features['aspect_ratio'] = 0.0
    
    # 3. Vertical and horizontal symmetry
    features['v_symmetry'] = compute_symmetry_score(binary_image, 'vertical')
    features['h_symmetry'] = compute_symmetry_score(binary_image, 'horizontal')
    
    # 4. Fill ratio (how much of bounding box is filled)
    total_pixels = binary_image.size
    white_pixels = cv2.countNonZero(binary_image)
    features['fill_ratio'] = white_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # 5. Top-heavy vs bottom-heavy
    mid = h // 2
    top_half = binary_image[:mid, :]
    bottom_half = binary_image[mid:, :]
    
    top_pixels = cv2.countNonZero(top_half)
    bottom_pixels = cv2.countNonZero(bottom_half)
    total = top_pixels + bottom_pixels
    
    if total > 0:
        features['top_heavy'] = top_pixels / total
        features['bottom_heavy'] = bottom_pixels / total
    else:
        features['top_heavy'] = 0.5
        features['bottom_heavy'] = 0.5
    
    return features


def validate_digit_topology(predicted_digit: int, features: Dict[str, float]) -> bool:
    """
    Validate if topological features match expected patterns for the predicted digit.
    
    Returns True if features are consistent, False if contradictory.
    """
    # Define expected characteristics for each digit
    rules = {
        1: {
            'holes': (0, 0),  # (min, max)
            'aspect_ratio': (1.8, 5.0),  # Tall and narrow
            'fill_ratio': (0.05, 0.25),  # Sparse
        },
        2: {
            'holes': (0, 0),
            'aspect_ratio': (0.8, 2.0),
            'fill_ratio': (0.15, 0.40),
        },
        3: {
            'holes': (0, 0),
            'aspect_ratio': (0.9, 2.0),
            'fill_ratio': (0.12, 0.35),
        },
        4: {
            'holes': (0, 1),  # Sometimes has a small hole
            'aspect_ratio': (0.9, 2.2),
            'fill_ratio': (0.10, 0.35),
        },
        5: {
            'holes': (0, 0),
            'aspect_ratio': (0.9, 2.0),
            'fill_ratio': (0.15, 0.38),
        },
        6: {
            'holes': (1, 1),  # Always has one hole
            'aspect_ratio': (1.0, 2.0),
            'bottom_heavy': (0.55, 1.0),  # Hole is at bottom
        },
        7: {
            'holes': (0, 0),
            'aspect_ratio': (0.9, 2.2),
            'top_heavy': (0.30, 0.60),  # Top bar
        },
        8: {
            'holes': (1, 2),  # Usually 2 holes, sometimes merged to 1
            'aspect_ratio': (0.8, 1.5),  # More square
            'v_symmetry': (0.6, 1.0),  # Vertically symmetric
        },
        9: {
            'holes': (1, 1),  # One hole
            'aspect_ratio': (1.0, 2.0),
            'top_heavy': (0.52, 1.0),  # Hole is at top
        },
        0: {
            'holes': (1, 1),  # One hole
            'aspect_ratio': (0.9, 1.6),
            'v_symmetry': (0.65, 1.0),  # Usually symmetric
        },
    }
    
    if predicted_digit not in rules:
        return True  # No rules defined, accept
    
    rule = rules[predicted_digit]
    violations = 0
    
    for feature_name, (min_val, max_val) in rule.items():
        if feature_name not in features:
            continue
        
        feature_val = features[feature_name]
        
        # Check if feature is outside expected range
        if feature_val < min_val or feature_val > max_val:
            violations += 1
    
    # Allow 1 violation, but not more (some tolerance for variation)
    return violations <= 1


def is_likely_empty_cell(binary_image: np.ndarray) -> bool:
    """
    Multi-factor empty cell detection to reduce false positives.
    
    Returns True if the cell is likely empty (noise/shadows, not a digit).
    """
    if binary_image.size == 0:
        return True
    
    # Factor 1: Variance check - empty cells have low variance
    variance = np.var(binary_image)
    if variance < 60:  # Increased from 30 - very uniform = likely empty
        return True
    
    # Factor 2: Edge density - real digits have clear edges
    edges = cv2.Canny(binary_image, 50, 150)
    edge_ratio = cv2.countNonZero(edges) / edges.size if edges.size > 0 else 0
    if edge_ratio < 0.012:  # Increased from 0.008 - very few edges = likely empty
        return True
    
    # Factor 3: Component fragmentation - too many small fragments = noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    if num_labels > 6:  # Too fragmented = noise, not digit
        return True
    
    # Factor 4: Perimeter-to-area ratio for largest component
    if num_labels > 1:
        # Find largest component (skip background)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        if areas:
            largest_idx = areas.index(max(areas)) + 1
            
            # Create mask for largest component
            component_mask = (labels == largest_idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                area = cv2.contourArea(contours[0])
                if area > 10:  # Avoid division by very small areas
                    compactness = perimeter * perimeter / (4 * np.pi * area)
                    if compactness > 8:  # Too irregular = noise
                        return True
    
    return False


def validate_detected_digit(binary_image: np.ndarray, detected_digit: int, score: float) -> bool:
    """
    Multi-factor validation to filter false positives.
    
    Combines:
    1. Score threshold (higher for prone-to-false-positive digits)
    2. Aspect ratio check (digit 1 should be tall and narrow)
    3. White pixel density (real digits have substantial fill)
    4. Connected component analysis (reject fragmented noise)
    
    Args:
        binary_image: The cell image with the detected digit
        detected_digit: The digit detected by template matching
        score: The template matching confidence score
    
    Returns:
        True if the digit passes validation, False if likely false positive
    """
    if detected_digit == 0:
        return True  # Empty cell, nothing to validate
    
    if binary_image.size == 0:
        return False
    
    h, w = binary_image.shape[:2]
    
    # === 1. Score threshold - RAISED for better filtering ===
    # Digits 1 and 4 need very high confidence
    if detected_digit == 1:
        min_score = 0.70
    elif detected_digit == 4:
        min_score = 0.72  # Even higher for 4
    else:
        min_score = 0.58  # Raised from 0.55
    
    if score < min_score:
        return False
    
    # === 2. White pixel density check ===
    white_pixels = cv2.countNonZero(binary_image)
    total_pixels = binary_image.size
    density = white_pixels / total_pixels if total_pixels > 0 else 0
    
    # Digits should have 4-35% fill
    if density < 0.04 or density > 0.35:
        return False
    
    # === 3. Connected component check - reject fragmented noise ===
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Too many components = fragmented noise
    if num_labels > 5:
        return False
    
    # Find largest component (skip background at index 0)
    if num_labels > 1:
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        largest_area = max(areas) if areas else 0
        total_white = sum(areas)
        
        # Largest component should be majority of white pixels (>60%)
        if total_white > 0 and largest_area / total_white < 0.6:
            return False
        
        # Largest component needs minimum size
        if largest_area < total_pixels * 0.02:
            return False
    
    # === 4. Aspect ratio check for digit 1 ===
    if detected_digit == 1:
        coords = cv2.findNonZero(binary_image)
        if coords is not None and len(coords) > 10:
            x, y, bw, bh = cv2.boundingRect(coords)
            aspect = bh / bw if bw > 0 else 1
            
            # Digit 1 should be tall and narrow (aspect ratio > 2.0)
            if aspect < 2.0:
                return False
            
            # Width should be narrow
            width_ratio = bw / w
            if width_ratio > 0.45:
                return False
    
    # === 5. Stricter validation for digit 4 ===
    if detected_digit == 4:
        coords = cv2.findNonZero(binary_image)
        if coords is not None and len(coords) > 10:
            x, y, bw, bh = cv2.boundingRect(coords)
            aspect = bh / bw if bw > 0 else 1
            
            # Digit 4: aspect ratio 1.2-2.3
            if aspect > 2.3 or aspect < 1.2:
                return False
            
            # Width at least 35% of cell
            width_ratio = bw / w
            if width_ratio < 0.35:
                return False
            
            # Height at least 50% of cell
            height_ratio = bh / h
            if height_ratio < 0.50:
                return False
    
    return True


def get_harris_corner_count(binary_image: np.ndarray) -> int:
    """
    Count Harris corners in a binary image.
    Useful for debugging and tuning thresholds.
    """
    if binary_image.size == 0:
        return 0
    
    gray = binary_image.astype(np.float32)
    harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    
    threshold = 0.01 * harris.max() if harris.max() > 0 else 0
    corners = harris > threshold
    
    return int(np.sum(corners))


def analyze_contour_features(binary_image: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Extract detailed contour-based shape descriptors.
    
    Returns dictionary of features or None if no contour found.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Use largest contour
    contour = max(contours, key=cv2.contourArea)
    
    if len(contour) < 5:  # Need at least 5 points for meaningful analysis
        return None
    
    features = {}
    
    # 1. Hu Moments (rotation/scale invariant shape descriptors)
    moments = cv2.moments(contour)
    if moments['m00'] != 0:  # Avoid division by zero
        hu_moments = cv2.HuMoments(moments).flatten()
        # Store first 3 Hu moments (most discriminative)
        for i in range(min(3, len(hu_moments))):
            features[f'hu_{i}'] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]) + 1e-10)
    
    # 2. Solidity (compactness measure)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    if hull_area > 0:
        features['solidity'] = contour_area / hull_area
    else:
        features['solidity'] = 0.0
    
    # 3. Convexity defects (concavities)
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if len(hull_indices) > 3 and len(contour) > 3:
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                features['num_defects'] = len(defects)
                # Depth of deepest defect
                features['max_defect_depth'] = max(defects[:, 0, 3]) / 256.0 if len(defects) > 0 else 0
            else:
                features['num_defects'] = 0
                features['max_defect_depth'] = 0.0
        except:
            features['num_defects'] = 0
            features['max_defect_depth'] = 0.0
    else:
        features['num_defects'] = 0
        features['max_defect_depth'] = 0.0
    
    # 4. Extent (ratio of contour area to bounding rectangle area)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    if rect_area > 0:
        features['extent'] = contour_area / rect_area
    else:
        features['extent'] = 0.0
    
    return features


def analyze_skeleton(binary_image: np.ndarray) -> Dict[str, int]:
    """
    Extract skeleton-based features (endpoints, junctions).
    
    Useful for distinguishing digits:
    - 1: 2 endpoints, 0 junctions
    - 7: 3 endpoints, 0 junctions  
    - 4: 4 endpoints, 1 junction
    - 8: 0 endpoints, 2+ junctions
    """
    features = {'num_endpoints': 0, 'num_junctions': 0}
    
    if binary_image.size == 0 or cv2.countNonZero(binary_image) == 0:
        return features
    
    # Skeletonize using Zhang-Suen thinning
    try:
        skeleton = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except:
        # Fallback if ximgproc not available
        return features
    
    endpoints = []
    junctions = []
    
    # Analyze skeleton pixels
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 0:
                continue
            
            # Count 8-connected neighbors
            neighbors = [
                skeleton[y-1, x-1], skeleton[y-1, x], skeleton[y-1, x+1],
                skeleton[y, x-1],                       skeleton[y, x+1],
                skeleton[y+1, x-1], skeleton[y+1, x], skeleton[y+1, x+1]
            ]
            neighbor_count = sum(1 for n in neighbors if n > 0)
            
            if neighbor_count == 1:
                endpoints.append((x, y))
            elif neighbor_count >= 3:
                junctions.append((x, y))
    
    features['num_endpoints'] = len(endpoints)
    features['num_junctions'] = len(junctions)
    
    return features


def validate_digit_with_advanced_features(predicted_digit: int, 
                                          contour_features: Optional[Dict[str, float]],
                                          skeleton_features: Dict[str, int]) -> bool:
    """
    Validate digit prediction using contour and skeleton features.
    
    Returns True if features are consistent with predicted digit.
    """
    if predicted_digit == 0:
        return True
    
    # Skeleton-based rules
    skeleton_rules = {
        1: {'endpoints': (2, 2), 'junctions': (0, 0)},
        7: {'endpoints': (2, 4), 'junctions': (0, 1)},
        4: {'endpoints': (3, 5), 'junctions': (1, 3)},
        8: {'endpoints': (0, 1), 'junctions': (1, 4)},
        0: {'endpoints': (0, 1), 'junctions': (0, 2)},
    }
    
    violations = 0
    
    # Check skeleton features
    if predicted_digit in skeleton_rules:
        rule = skeleton_rules[predicted_digit]
        for feature_name, (min_val, max_val) in rule.items():
            if feature_name == 'endpoints':
                val = skeleton_features.get('num_endpoints', 0)
            elif feature_name == 'junctions':
                val = skeleton_features.get('num_junctions', 0)
            else:
                continue
            
            if val < min_val or val > max_val:
                violations += 1
    
    # Check contour features
    if contour_features:
        contour_rules = {
            1: {'solidity': (0.7, 1.0), 'num_defects': (0, 2)},
            8: {'solidity': (0.4, 0.75), 'num_defects': (2, 10)},
            6: {'solidity': (0.55, 0.85), 'num_defects': (1, 6)},
            9: {'solidity': (0.55, 0.85), 'num_defects': (1, 6)},
            0: {'solidity': (0.6, 0.9), 'num_defects': (0, 4)},
        }
        
        if predicted_digit in contour_rules:
            rule = contour_rules[predicted_digit]
            for feature_name, (min_val, max_val) in rule.items():
                val = contour_features.get(feature_name, 0)
                if val < min_val or val > max_val:
                    violations += 1
    
    # Allow up to 1 violation for tolerance
    return violations <= 1



def _extract_digit_with_sliding_window(cell: np.ndarray, templates: Dict[int, List[np.ndarray]],
                                       step_sizes: List[float] = [0.4, 1.0],
                                       window_scales: List[float] = [0.7, 0.85, 1.0]) -> Tuple[int, float, List[Tuple[int, float]]]:
    """
    Extract digit using enhanced sliding window with multiple window sizes and positions.
    
    Fast version with minimal configurations but better coverage than the original.
    Key changes from original:
    - More window scales (0.7, 0.85, 1.0) vs (0.7, 0.85, 1.0, 1.15)
    - Fewer step sizes (0.4, 1.0) - medium and full step only
    - No aspect ratio variations
    - Voting-based candidate selection
    
    Args:
        cell: Input cell image
        templates: Digit templates
        step_sizes: List of step sizes as fraction of window size
        window_scales: List of scales relative to cell size
    
    Returns:
        (best_digit, best_score, candidates)
    """
    h, w = cell.shape
    
    best_digit = 0
    best_score = 0.0
    all_results = {}  # digit -> list of scores
    
    # Simple window configurations based on scales only
    window_configs = []
    for scale in window_scales:
        win_size = int(min(h, w) * scale)
        win_size = min(win_size, min(h, w))
        if win_size >= 15:
            window_configs.append(win_size)
    
    # Remove duplicates
    window_configs = list(set(window_configs))
    
    # Process each window size
    for win_size in window_configs:
        for step_frac in step_sizes:
            step = max(2, int(win_size * step_frac))
            
            # Key positions: corners, center, and stepped positions
            y_positions = [0]  # Top
            x_positions = [0]  # Left
            
            # Center
            center_y = (h - win_size) // 2
            center_x = (w - win_size) // 2
            
            # Edge
            edge_y = max(0, h - win_size)
            edge_x = max(0, w - win_size)
            
            # Add stepped positions if step_frac < 1.0
            if step_frac < 1.0:
                for y in range(step, edge_y, step):
                    y_positions.append(y)
                for x in range(step, edge_x, step):
                    x_positions.append(x)
            
            # Add center and edges
            y_positions.extend([center_y, edge_y])
            x_positions.extend([center_x, edge_x])
            
            # Remove duplicates and filter valid
            y_positions = sorted(set(p for p in y_positions if 0 <= p <= h - win_size))
            x_positions = sorted(set(p for p in x_positions if 0 <= p <= w - win_size))
            
            for y in y_positions:
                for x in x_positions:
                    window = cell[y:y+win_size, x:x+win_size]
                    digit, score, _ = _extract_digit_from_cell_core(window, templates)
                    
                    if digit > 0 and score > 0.50:  # Raised threshold to reduce false positives
                        if digit not in all_results:
                            all_results[digit] = []
                        all_results[digit].append(score)
                        
                        if score > best_score:
                            best_score = score
                            best_digit = digit
    
    # Build candidate list with voting
    candidates = []
    digit_stats = {}
    
    for digit, scores in all_results.items():
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        count = len(scores)
        
        # Skip if max score is too low (likely noise)
        if max_score < 0.50:
            continue
        
        # Combined score: max score + bonus for consistency
        combined = max_score * 0.85 + avg_score * 0.1 + min(count / 10.0, 0.05)
        digit_stats[digit] = {'max': max_score, 'avg': avg_score, 'count': count, 'combined': combined}
    
    sorted_digits = sorted(digit_stats.keys(), key=lambda d: digit_stats[d]['combined'], reverse=True)
    
    for digit in sorted_digits[:3]:
        candidates.append((digit, digit_stats[digit]['max']))
    
    # Use voting winner if max_score is not highly confident
    if best_score < 0.55 and sorted_digits:
        if digit_stats[sorted_digits[0]]['combined'] > digit_stats.get(best_digit, {}).get('combined', 0):
            best_digit = sorted_digits[0]
            best_score = digit_stats[sorted_digits[0]]['max']
    
    # Multi-factor validation - reject false positives
    if best_digit > 0:
        if not validate_detected_digit(cell, best_digit, best_score):
            # Failed validation - likely false positive
            # Try next best candidate
            for candidate in candidates:
                cand_digit, cand_score = candidate
                if cand_digit != best_digit and cand_digit != 0:
                    if validate_detected_digit(cell, cand_digit, cand_score):
                        best_digit = cand_digit
                        best_score = cand_score
                        break
            else:
                # No valid candidate - mark as empty
                best_digit = 0
                best_score = 0.0
    
    return best_digit, best_score, candidates


def _extract_digit_from_cell(cell: np.ndarray, templates: Dict[int, List[np.ndarray]],
                             min_fill: float = 0.004,
                             accept_threshold: float = 0.48,
                             use_sliding_window: bool = True) -> Tuple[int, float, List[Tuple[int, float]]]:
    """
    Recognize a digit from a single cell image using enhanced preprocessing.
    
    Args:
        use_sliding_window: If True, use sliding window multi-scale detection

    Returns:
        (digit, score, candidates) where digit=0 means empty cell.
    """
    if use_sliding_window:
        # Try sliding window approach first
        return _extract_digit_with_sliding_window(cell, templates)
    else:
        # Use original single-extraction approach
        return _extract_digit_from_cell_core(cell, templates, min_fill, accept_threshold)


def _extract_digit_from_cell_core(cell: np.ndarray, templates: Dict[int, List[np.ndarray]],
                                  min_fill: float = 0.004,
                                  accept_threshold: float = 0.48) -> Tuple[int, float, List[Tuple[int, float]]]:
    """
    Core digit extraction logic (original implementation).
    Now called by wrapper functions that can use sliding window or direct extraction.

    Returns:
        (digit, score, candidates) where digit=0 means empty cell.
    """
    h, w = cell.shape
    margin = max(1, int(min(h, w) * 0.05))
    inner = cell[margin:h - margin, margin:w - margin]

    # Normalize polarity to make digits white on black
    work = inner.copy()
    if work.mean() > 127:
        work = cv2.bitwise_not(work)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This improves contrast for digits in varying lighting conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    work = clahe.apply(work)
    
    # Bilateral filter: reduces noise while preserving edges
    work = cv2.bilateralFilter(work, 5, 50, 50)
    
    # Try multiple adaptive thresholding strategies and pick the best
    binary_candidates = []
    
    # Method 1: Otsu thresholding
    work_blur = cv2.GaussianBlur(work, (3, 3), 0)
    _, binary1 = cv2.threshold(work_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_candidates.append(binary1)
    
    # Method 2: Adaptive threshold with smaller block size
    binary2 = cv2.adaptiveThreshold(work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    binary_candidates.append(binary2)
    
    # Method 3: Adaptive threshold with larger block size
    block_size = min(work.shape[0], work.shape[1]) // 3
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(11, min(31, block_size))
    binary3 = cv2.adaptiveThreshold(work, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block_size, 2)
    binary_candidates.append(binary3)
    
    # Select best binarization based on quality heuristics
    best_binary = None
    best_quality = -1
    
    for binary in binary_candidates:
        # Suppress residual grid bleed by clearing borders
        test_binary = binary.copy()
        border = max(2, int(min(h, w) * 0.08))
        test_binary[:border, :] = 0
        test_binary[-border:, :] = 0
        test_binary[:, :border] = 0
        test_binary[:, -border:] = 0
        
        # Quality = non-zero pixels in reasonable range
        non_zero = cv2.countNonZero(test_binary)
        total = test_binary.size
        fill_ratio = non_zero / total
        
        # Good digits have 8-35% fill (more restrictive to reduce noise)
        if 0.08 <= fill_ratio <= 0.35:
            quality = 1.0 - abs(0.18 - fill_ratio)  # Prefer ~18% fill
            if quality > best_quality:
                best_quality = quality
                best_binary = test_binary
    
    if best_binary is None:
        # Fallback to first candidate
        binary = binary_candidates[0].copy()
        border = max(2, int(min(h, w) * 0.08))
        binary[:border, :] = 0
        binary[-border:, :] = 0
        binary[:, :border] = 0
        binary[:, -border:] = 0
        best_binary = binary

    binary = best_binary
    
    # NOTE: Empty cell detection disabled - was too aggressive
    # Caused regression from Iteration 2 (broke 01.jpg and 03.jpg)
    # if is_likely_empty_cell(binary):
    #     return 0, 0.0, []
    
    # Morphological operations to clean up digit
    # Opening: remove small noise
    kernel_small = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    
    # Closing: fill small gaps in digits
    kernel_med = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_med)

    # Connected components to isolate the digit blob
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0, 0.0, []

    # Filter components by area and aspect ratio
    valid_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        
        min_area = min_fill * (binary.shape[0] * binary.shape[1])
        max_area = 0.7 * binary.size
        
        if area < min_area or area > max_area:
            continue
            
        # Check aspect ratio (digits are typically 1:1 to 1:3)
        aspect_ratio = bh / max(bw, 1)
        if aspect_ratio < 0.5 or aspect_ratio > 5.0:
            continue
            
        valid_components.append((i, area))
    
    if not valid_components:
        return 0, 0.0, []
    
    # Select largest valid component
    digit_label = max(valid_components, key=lambda x: x[1])[0]
    x, y, bw, bh = (stats[digit_label, cv2.CC_STAT_LEFT], stats[digit_label, cv2.CC_STAT_TOP],
                    stats[digit_label, cv2.CC_STAT_WIDTH], stats[digit_label, cv2.CC_STAT_HEIGHT])

    # Extract digit ROI with padding
    pad = 2
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(binary.shape[1], x + bw + pad), min(binary.shape[0], y + bh + pad)
    digit_roi = binary[y1:y2, x1:x2]

    # Square-pad before resize to maintain aspect ratio
    side = max(digit_roi.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - digit_roi.shape[0]) // 2
    x_off = (side - digit_roi.shape[1]) // 2
    square[y_off:y_off + digit_roi.shape[0], x_off:x_off + digit_roi.shape[1]] = digit_roi

    # Resize to template size
    target_size = next(iter(templates.values()))[0].shape[0]
    digit_norm = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    digit_norm = cv2.normalize(digit_norm, None, 0, 255, cv2.NORM_MINMAX)

    # Try both normal and inverted ROI for robustness
    candidates = [digit_norm, cv2.bitwise_not(digit_norm)]

    # Multi-method template matching with ensemble voting
    best_digit, best_score = 0, -1.0
    digit_votes: Dict[int, List[float]] = {d: [] for d in range(1, 10)}
    
    for candidate in candidates:
        for digit, tmpl_list in templates.items():
            scores_for_digit = []
            
            for tmpl in tmpl_list:
                # Try multiple matching methods
                score_ccoeff = cv2.matchTemplate(candidate, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
                score_ccorr = cv2.matchTemplate(candidate, tmpl, cv2.TM_CCORR_NORMED)[0][0]
                
                # Ensemble: weight CCOEFF more heavily
                ensemble_score = 0.7 * score_ccoeff + 0.3 * score_ccorr
                scores_for_digit.append(ensemble_score)
            
            # Take max score across all templates for this digit
            if scores_for_digit:
                max_score = max(scores_for_digit)
                digit_votes[digit].append(max_score)
    
    # Aggregate votes: use max score for each digit
    for digit, scores in digit_votes.items():
        if scores:
            avg_score = max(scores)
            if avg_score > best_score:
                best_digit, best_score = digit, float(avg_score)

    # Check if top match is significantly better than second-best
    sorted_scores = sorted([(d, max(scores)) for d, scores in digit_votes.items() if scores],
                          key=lambda x: x[1], reverse=True)
    
    if len(sorted_scores) >= 2:
        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        confidence_margin = top_score - second_score
        
        # If top and second are too close, be more conservative (increased threshold)
        if confidence_margin < 0.15:
            accept_threshold = max(accept_threshold, 0.55)
    
    # NEW: Topological validation (from Iteration 1)
    # Analyze structural features of the digit
    features = analyze_digit_topology(digit_norm)
    
    # NOTE: Advanced features disabled to restore Iteration 2 functionality
    # contour_features = analyze_contour_features(digit_norm)
    # skeleton_features = analyze_skeleton(digit_norm)
    
    # Validate if best_digit matches topological features
    if best_digit > 0 and not validate_digit_topology(best_digit, features):
        # Topology doesn't match! Try second-best digit
        if len(sorted_scores) >= 2:
            second_digit = sorted_scores[1][0]
            second_score_val = sorted_scores[1][1]
            
            # Check if second best matches topology
            topology_valid_2 = validate_digit_topology(second_digit, features)
            
            if topology_valid_2:
                # Use second best instead
                best_digit = second_digit
                best_score = second_score_val
            else:
                # Neither matches well - lower confidence but keep best
                best_score *= 0.7
    
    # NOTE: Tesseract disabled for now - not affecting core functionality
    tesseract_digit = 0
    
    # Prepare candidates list (top 3) - simplified
    final_candidates = [(best_digit, best_score)] if best_digit > 0 else []
    
    # Add runner-ups
    for d, s in sorted_scores[:3]:  # Take top 3
        if d != best_digit and d > 0:
            final_candidates.append((d, s))
            if len(final_candidates) >= 3:
                break
    
    # Digit-specific threshold adjustment
    # Digits 4 and 8 are commonly false positives (empty cells, noise)
    # Digit 8 is the worst offender - require very high confidence
    effective_threshold = accept_threshold
    if best_digit == 8:
        effective_threshold = accept_threshold * 1.4  # 40% higher (0.48 -> 0.67)
    elif best_digit == 4:
        effective_threshold = accept_threshold * 1.25  # 25% higher (0.48 -> 0.60)
    
    # Apply threshold with 5% leniency
    if best_score < effective_threshold * 0.95:
        return 0, best_score, []

    return best_digit, best_score, final_candidates



def resolve_conflicts(board: np.ndarray, scores: np.ndarray, min_score_keep: float = 0.0
                      ) -> Tuple[np.ndarray, List[str]]:
    """
    Remove duplicate givens in rows/cols/blocks by keeping only the highest-score hit.

    Returns:
        cleaned_board, list of notes about removed cells
    """
    cleaned = board.copy()
    notes: List[str] = []

    def drop_duplicates(index_list: List[Tuple[int, int]], label: str):
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


def extract_grid_digits(binary_grid: np.ndarray, templates: Optional[Dict[int, List[np.ndarray]]] = None,
                        accept_threshold: float = 0.45
                        ) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], List[Tuple[int, float]]]]:
    """
    Extract a 9x9 integer grid (0 = empty) from the straightened binary image.

    Returns:
        board (9x9 ints), scores (9x9 floats), all_candidates (dict)
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
    all_candidates = {}

    for r in range(9):
        for c in range(9):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            cell1 = digits_only[y1:y2, x1:x2]
            cell2 = raw_view[y1:y2, x1:x2]
            
            d1, s1, c1 = _extract_digit_from_cell(cell1, templates, accept_threshold=accept_threshold)
            d2, s2, c2 = _extract_digit_from_cell(cell2, templates, accept_threshold=accept_threshold)
            
            if s2 > s1:
                digit, score, candidates = d2, s2, c2
            else:
                digit, score, candidates = d1, s1, c1
                
            board[r, c] = digit
            scores[r, c] = score
            all_candidates[(r, c)] = candidates

    return board, scores, all_candidates


def validate_and_correct_ocr(board: np.ndarray, scores: np.ndarray, 
                              all_candidates: Optional[Dict[Tuple[int, int], List[Tuple[int, float]]]] = None
                              ) -> Tuple[np.ndarray, List[str]]:
    """
    Use Sudoku constraints to validate and correct OCR results.
    
    Args:
        board: 9x9 numpy array of detected digits
        scores: 9x9 numpy array of confidence scores  
        all_candidates: Optional dict mapping (row, col) -> list of (digit, score) alternatives
        
    Returns:
        corrected_board, list of correction notes
    """
    from .solver import _is_valid
    
    corrections = []
    working_board = board.copy()
    
    # Check each cell for Sudoku constraint violations
    for r in range(9):
        for c in range(9):
            digit = working_board[r, c]
            if digit == 0:
                continue
            
            # Temporarily remove digit to check if it's valid
            working_board[r, c] = 0
            
            # Check if placing it back is valid
            if _is_valid(working_board, r, c, digit):
                # Valid, restore it
                working_board[r, c] = digit
            else:
                # Conflict detected!
                corrections.append(f"Conflict at ({r+1},{c+1}): {digit} violates Sudoku rules")
                
                # Try to find valid alternative from candidates
                found_alternative = False
                if all_candidates and (r, c) in all_candidates:
                    candidates = all_candidates[(r, c)]
                    # Candidates are already sorted by score
                    for cand_digit, cand_score in candidates:
                        if cand_digit == digit:
                            continue # Skip the one we just failed
                        
                        if _is_valid(working_board, r, c, cand_digit):
                            # Found a valid alternative!
                            working_board[r, c] = cand_digit
                            corrections.append(f"  → Swapped invalid {digit} for valid candidate {cand_digit} (score {cand_score:.2f})")
                            found_alternative = True
                            break
                
                if not found_alternative:
                    corrections.append(f"  → Removed invalid digit {digit} at ({r+1},{c+1}) (no valid candidates)")
                    # Leave cell as 0 (empty)
    
    return working_board, corrections


def extract_grid_digits_with_validation(binary_grid: np.ndarray, 
                                        templates: Optional[Dict[int, List[np.ndarray]]] = None,
                                        accept_threshold: float = 0.45
                                        ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract digits and apply puzzle-level validation.
    
    Returns:
        board (9x9 ints), scores (9x9 floats), validation_notes (list of strings)
    """
    # First pass: extract all digits
    board, scores, all_candidates = extract_grid_digits(binary_grid, templates, accept_threshold=accept_threshold)
    
    # Apply puzzle-level validation
    validated_board, validation_notes = validate_and_correct_ocr(board, scores, all_candidates)
    
    return validated_board, scores, validation_notes




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


def render_ocr_detection(image: np.ndarray, board: np.ndarray, scores: np.ndarray,
                         show_confidence: bool = True) -> np.ndarray:
    """
    Overlay OCR detection results on the straightened image with confidence visualization.
    
    Colors indicate confidence level:
    - Green (>70%): High confidence detection
    - Yellow/Orange (50-70%): Medium confidence detection  
    - Red (<50%): Low confidence detection
    - Empty cells show small X marker
    
    Args:
        image: Straightened grid image
        board: 9x9 numpy array of detected digits (0 = empty)
        scores: 9x9 numpy array of confidence scores (0.0 to 1.0)
        show_confidence: If True, show confidence percentage below digit
    
    Returns:
        Annotated image with OCR detections visualized
    """
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    h, w = canvas.shape[:2]
    cell_h = h // 9
    cell_w = w // 9
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw grid lines for clarity
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        y = int(i * cell_h)
        cv2.line(canvas, (0, y), (w, y), (80, 80, 80), thickness)
        x = int(i * cell_w)
        cv2.line(canvas, (x, 0), (x, h), (80, 80, 80), thickness)

    for r in range(9):
        for c in range(9):
            val = int(board[r, c])
            score = float(scores[r, c])
            
            cx = c * cell_w + cell_w // 2
            cy = r * cell_h + cell_h // 2
            
            if val == 0:
                # Empty cell - draw small X
                size = 5
                cv2.line(canvas, (cx - size, cy - size), (cx + size, cy + size), (100, 100, 100), 1)
                cv2.line(canvas, (cx - size, cy + size), (cx + size, cy - size), (100, 100, 100), 1)
                continue
            
            # Color based on confidence
            if score >= 0.70:
                color = (0, 255, 0)       # Green - high confidence
            elif score >= 0.50:
                color = (0, 200, 255)     # Orange - medium confidence
            else:
                color = (0, 0, 255)       # Red - low confidence
            
            # Draw digit
            text = str(val)
            font_scale = 1.0
            thickness = 2
            size_text, _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            tx = cx - size_text[0] // 2
            ty = cy + size_text[1] // 2 - (5 if show_confidence else 0)
            
            # Draw text with black outline
            cv2.putText(canvas, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (tx, ty), font, font_scale, color, thickness, cv2.LINE_AA)
            
            if show_confidence:
                conf_text = f"{int(score * 100)}%"
                conf_size, _ = cv2.getTextSize(conf_text, font, 0.35, 1)
                conf_x = cx - conf_size[0] // 2
                conf_y = cy + cell_h // 3
                cv2.putText(canvas, conf_text, (conf_x, conf_y), font, 0.35, color, 1, cv2.LINE_AA)

    # Legend
    legend_y = h - 10
    cv2.putText(canvas, "High", (10, legend_y), font, 0.35, (0, 255, 0), 1)
    cv2.putText(canvas, "Med", (55, legend_y), font, 0.35, (0, 200, 255), 1)
    cv2.putText(canvas, "Low", (95, legend_y), font, 0.35, (0, 0, 255), 1)

    return canvas

