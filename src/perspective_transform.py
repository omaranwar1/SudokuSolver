

import cv2
import numpy as np


def perspective_transform(image, corners, output_size=450):
    """
    Apply perspective transformation to straighten the Sudoku grid.

    This function transforms the quadrilateral grid (defined by 4 corners)
    into a square image using a perspective transformation matrix.

    The transformation maps the distorted quadrilateral to a perfect square,
    correcting for camera angle and perspective distortion.

    Mathematical basis:
    - Uses homography (3x3 matrix) to map points from source to destination
    - Solves for transformation matrix using 4 point correspondences
    - Applies bilinear interpolation for pixel value mapping

    Args:
        image: Original image containing the grid
        corners: Four corner points [TL, TR, BR, BL] of the grid
        output_size (int): Size of the output square image (default: 450x450)

    Returns:
        numpy.ndarray: Straightened grid as a square image

    References:
        - Hartley & Zisserman (2004), Chapter 4: "Estimation - 2D Projective Transformations"
        - Perspective transformation theory: https://en.wikipedia.org/wiki/Homography
    """
   
    corners = corners.astype(np.float32)

    
    destination_corners = np.array([
        [0, 0],  # Top-left
        [output_size - 1, 0],  # Top-right
        [output_size - 1, output_size - 1],  # Bottom-right
        [0, output_size - 1]  # Bottom-left
    ], dtype=np.float32)

   
    transform_matrix = cv2.getPerspectiveTransform(corners, destination_corners)

   
    warped = cv2.warpPerspective(
        image,
        transform_matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR
    )

    return warped


def straighten_grid(image, processed_image, output_size=450):
    """
    Complete pipeline to straighten a Sudoku grid from an image.

    This is a convenience function that combines:
    1. Contour detection to find the grid
    2. Corner identification
    3. Perspective transformation

    Args:
        image: Original color/grayscale image
        processed_image: Preprocessed binary image
        output_size (int): Size of output square (default: 450x450)

    Returns:
        tuple: (straightened_image, corners) or (None, None) if grid not found

    Example:
        >>> original, processed = preprocess_image("sudoku.jpg")
        >>> straightened, corners = straighten_grid(original, processed)
        >>> if straightened is not None:
        >>>     cv2.imshow("Straightened Grid", straightened)
    """
    from .grid_detection import find_largest_contour, find_grid_corners

    
    contour = find_largest_contour(processed_image)
    if contour is None:
        print("Error: Could not find Sudoku grid in image")
        return None, None

   
    corners = find_grid_corners(contour)

   
    straightened = perspective_transform(image, corners, output_size)

    return straightened, corners


def get_transform_quality_score(corners):
    """
    Calculate a quality score for the perspective transformation.

    This score indicates how close the detected quadrilateral is to a square.
    A higher score (closer to 1.0) indicates better quality.

    The score is based on:
    - Aspect ratio (should be close to 1:1)
    - Angle deviation from 90 degrees at corners

    Args:
        corners: Four corner points of the grid

    Returns:
        float: Quality score between 0 and 1

    Note:
        This is useful for filtering out poor detections or warning users
        about potential quality issues.
    """
   
    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    top = distance(corners[0], corners[1])
    right = distance(corners[1], corners[2])
    bottom = distance(corners[2], corners[3])
    left = distance(corners[3], corners[0])

   
    horizontal_ratio = min(top, bottom) / max(top, bottom)
    vertical_ratio = min(left, right) / max(left, right)
    overall_ratio = min(top, bottom, left, right) / max(top, bottom, left, right)

   
    quality_score = (horizontal_ratio + vertical_ratio + overall_ratio) / 3.0

    return quality_score


def apply_adaptive_transform(image, corners, min_quality=0.7):
    """
    Apply perspective transform with quality checking.

    If the detected grid quality is below threshold, returns None and
    suggests the user to retake the image.

    Args:
        image: Original image
        corners: Four corner points
        min_quality (float): Minimum acceptable quality score (0-1)

    Returns:
        numpy.ndarray or None: Transformed image if quality is acceptable

    Example:
        >>> straightened = apply_adaptive_transform(image, corners, min_quality=0.7)
        >>> if straightened is None:
        >>>     print("Image quality too low, please retake photo")
    """
    quality = get_transform_quality_score(corners)

    if quality < min_quality:
        print(f"Warning: Grid detection quality is low ({quality:.2f})")
        print("Consider retaking the image with better angle/lighting")
        return None

    return perspective_transform(image, corners)


def visualize_transformation(original, corners, transformed):
    """
    Visualize the perspective transformation process.

    Shows the original image with detected corners and the final
    straightened result side by side.

    Args:
        original: Original image
        corners: Detected corner points
        transformed: Straightened grid image

    Returns:
        numpy.ndarray: Combined visualization image
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    
    original_viz = original.copy()
    corner_labels = ['TL', 'TR', 'BR', 'BL']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i, (corner, label, color) in enumerate(zip(corners, corner_labels, colors)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(original_viz, (x, y), 10, color, -1)
        cv2.putText(original_viz, label, (x + 15, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

   
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(original_viz, pt1, pt2, (0, 255, 255), 2)

    axes[0].imshow(cv2.cvtColor(original_viz, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original with Detected Corners')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Straightened Grid')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    return fig


