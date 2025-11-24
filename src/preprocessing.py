"""Image preprocessing utilities for Sudoku grid extraction."""


import cv2
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def detect_noise_type(image: np.ndarray) -> str:
    """
    Detect the type of noise in the image.

    Returns:
        'gaussian' for Gaussian (additive white) noise
        'salt_pepper' for salt-and-pepper noise
    """
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    noise = image.astype(np.float32) - smoothed.astype(np.float32)
    noise_mean = np.abs(np.mean(noise))

    total_pixels = image.size
    extreme_dark = np.sum(image < 10) / total_pixels
    extreme_bright = np.sum(image > 245) / total_pixels
    extreme_ratio = extreme_dark + extreme_bright

    if noise_mean < 5 and extreme_ratio < 0.05:
        return 'gaussian'
    elif extreme_ratio > 0.10:
        return 'salt_pepper'
    else:
        return 'gaussian'


def remove_gaussian_noise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """Remove Gaussian noise with adaptive strength."""
    print(f"  Noise type: Gaussian (additive white)")

    if noise_level > 40000:
        print("  -> ULTRA-HIGH Gaussian noise - HEAVY denoising BEFORE binarization")

        result = cv2.fastNlMeansDenoising(image, None, h=40, templateWindowSize=7, searchWindowSize=21)

        result = cv2.fastNlMeansDenoising(result, None, h=25, templateWindowSize=7, searchWindowSize=21)

        result = cv2.bilateralFilter(result, 11, 100, 100)

    elif noise_level > 15000:
        print("  -> HIGH Gaussian noise - strong NLM denoising")

        result = cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)
        result = cv2.bilateralFilter(result, 7, 50, 50)

    elif noise_level > 5000:
        print("  -> MODERATE Gaussian noise - balanced NLM denoising")

        result = cv2.fastNlMeansDenoising(image, None, h=20, templateWindowSize=7, searchWindowSize=21)
        result = cv2.bilateralFilter(result, 5, 40, 40)

    elif noise_level > 1000:
        print("  -> LOW Gaussian noise - light NLM denoising")

        result = cv2.fastNlMeansDenoising(image, None, h=15, templateWindowSize=7, searchWindowSize=21)

    else:
        print("  -> MINIMAL noise - bilateral filter only")
        result = cv2.bilateralFilter(image, 9, 75, 75)

    return result


def remove_salt_pepper_noise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """Remove salt-and-pepper noise with adaptive median filtering."""
    print(f"  Noise type: Salt-and-pepper")

    if noise_level > 50000:
        print("  -> ULTRA-HIGH salt-and-pepper - strong median filtering")

        result = cv2.medianBlur(image, 5)
        result = cv2.medianBlur(result, 5)
        result = cv2.medianBlur(result, 7)
        result = cv2.medianBlur(result, 5)

    elif noise_level > 15000:
        print("  -> HIGH salt-and-pepper - moderate median filtering")

        result = cv2.medianBlur(image, 5)
        result = cv2.medianBlur(result, 5)
        result = cv2.medianBlur(result, 3)

    elif noise_level > 5000:
        print("  -> MODERATE salt-and-pepper - light median filtering")

        result = cv2.medianBlur(image, 5)
        result = cv2.medianBlur(result, 3)

    elif noise_level > 1000:
        print("  -> LOW salt-and-pepper - minimal median")
        result = cv2.medianBlur(image, 5)

    else:
        print("  -> MINIMAL noise - bilateral filter")
        result = cv2.bilateralFilter(image, 9, 75, 75)

    return result


def remove_noise(image: np.ndarray) -> np.ndarray:
    """
    Intelligently remove noise by detecting type and applying appropriate strategy.

    Detects noise type (Gaussian vs salt-and-pepper) and applies optimal denoising.

    Returns:
        Denoised image
    """
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    print(f"  Noise level: {laplacian_var:.1f}")

    noise_type = detect_noise_type(image)

    if noise_type == 'gaussian':
        return remove_gaussian_noise(image, laplacian_var)
    else:
        return remove_salt_pepper_noise(image, laplacian_var)



def local_brightness_equalization(image: np.ndarray, strength: float = 0.7) -> np.ndarray:
    """
    Equalize brightness locally using flat-field correction.

    This method estimates the background illumination and removes it,
    creating more uniform brightness across the image.

    Args:
        image: Grayscale image
        strength: Correction strength 0-1 (default 0.7)

    Returns:
        Image with equalized local brightness
    """
    img_float = image.astype(np.float32)

    kernel_size = max(image.shape) // 8
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd
    kernel_size = max(51, min(kernel_size, 201))

    background = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), 0)

    target = np.median(img_float)

    correction = (target - background) * strength

    result = img_float + correction
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """
    Normalize brightness - bring all images to similar brightness range.
    """
    mean_val = np.mean(image)
    std_val = np.std(image)

    print(f"  Brightness: mean={mean_val:.1f}, std={std_val:.1f}")

    if mean_val < 50:
        print("  -> Extremely dark - strong gamma correction (γ=3.5)")
        gamma = 3.5
    elif mean_val < 100:
        print("  -> Very dark - moderate gamma correction (γ=2.5)")
        gamma = 2.5
    elif mean_val > 200:
        print("  -> Too bright - darkening (γ=0.6)")
        gamma = 0.6
    else:
        gamma = 1.0

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        result = cv2.LUT(image, table)
    else:
        result = image

    if gamma > 2.0:  # Only for very dark images with aggressive gamma
        print("  -> Applying local equalization after gamma")
        result = local_brightness_equalization(result, strength=0.7)

    if std_val < 40:
        print("  -> Low contrast - aggressive CLAHE")
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        result = clahe.apply(result)
    elif std_val < 60:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        result = clahe.apply(result)

    result = cv2.normalize(result, None, 20, 235, cv2.NORM_MINMAX)

    return result


def detect_rotation_robust(binary_image: np.ndarray) -> Optional[float]:
    """
    Detect rotation using cleaned binary image.
    Only returns rotation if VERY confident.
    """
    edges = cv2.Canny(binary_image, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                             minLineLength=binary_image.shape[1]//4, maxLineGap=20)

    if lines is None or len(lines) < 15:
        return None

    h_angles = []
    v_angles = []

    for line in lines[:60]:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 != 0:
            angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
        else:
            angle = 90

        angle = angle % 180

        if angle < 15 or angle > 165:  # Horizontal
            if angle > 165:
                h_angles.append(angle - 180)
            else:
                h_angles.append(angle)
        elif 75 < angle < 105:  # Vertical
            v_angles.append(angle - 90)

    if len(h_angles) < 5 or len(v_angles) < 5:
        return None

    h_median = np.median(h_angles)
    v_median = np.median(v_angles)

    if abs(h_median - v_median) > 2:
        return None

    rotation = (h_median + v_median) / 2

    if abs(rotation) > 4:
        return rotation

    return None


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Apply rotation correction."""
    h, w = image.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def analyze_histogram_quality(image: np.ndarray) -> Tuple[bool, float]:
    """
    Analyze if histogram has good bimodal separation.

    Returns:
        tuple: (is_good_separation, quality_score)
            - is_good_separation: True if histogram shows clear bimodal distribution
            - quality_score: 0-1 score indicating separation quality
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    from scipy.ndimage import gaussian_filter1d
    hist_smooth = gaussian_filter1d(hist, sigma=2)

    hist_smooth = hist_smooth / hist_smooth.sum()

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_threshold = _

    total_pixels = image.size
    weight_bg = np.sum(image < otsu_threshold) / total_pixels
    weight_fg = 1 - weight_bg

    if weight_bg == 0 or weight_fg == 0:
        return False, 0.0

    mean_bg = np.mean(image[image < otsu_threshold]) if weight_bg > 0 else 0
    mean_fg = np.mean(image[image >= otsu_threshold]) if weight_fg > 0 else 255

    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    quality_score = min(between_var / 3000.0, 1.0)

    from scipy.signal import find_peaks
    peaks, properties = find_peaks(hist_smooth, height=0.001, distance=20)

    has_two_peaks = len(peaks) >= 2
    is_good = has_two_peaks and quality_score > 0.3

    return is_good, quality_score


def adaptive_threshold_subimages(image: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """
    Apply adaptive thresholding using subimage analysis.

    Args:
        image: Grayscale image (8-bit)
        grid_size: Number of divisions per dimension (default 4 = 4x4 grid)

    Returns:
        Binary thresholded image
    """
    h, w = image.shape

    sub_h = h // grid_size
    sub_w = w // grid_size

    result = np.zeros_like(image)

    print(f"  Analyzing {grid_size}x{grid_size} subimages...")

    good_regions = 0
    poor_regions = 0

    for i in range(grid_size):
        for j in range(grid_size):
            y1 = i * sub_h
            y2 = (i + 1) * sub_h if i < grid_size - 1 else h
            x1 = j * sub_w
            x2 = (j + 1) * sub_w if j < grid_size - 1 else w

            subimg = image[y1:y2, x1:x2]

            is_good, quality = analyze_histogram_quality(subimg)

            if is_good:
                _, thresh_sub = cv2.threshold(subimg, 0, 255,
                                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                good_regions += 1
            else:
                thresh_sub = cv2.adaptiveThreshold(
                    subimg, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    15, 3
                )
                poor_regions += 1

            result[y1:y2, x1:x2] = thresh_sub

    print(f"  -> {good_regions} regions with good separation (Otsu)")
    print(f"  -> {poor_regions} regions with poor separation (Adaptive)")

    result = cv2.medianBlur(result, 3)

    return result


def complete_grid_lines(binary_image: np.ndarray) -> np.ndarray:
    """
    Complete disconnected grid lines using conservative morphology.
    """
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    h_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_h)
    v_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_v)

    h_lines = cv2.dilate(h_lines, kernel_h, iterations=1)
    v_lines = cv2.dilate(v_lines, kernel_v, iterations=1)

    lines = cv2.add(h_lines, v_lines)

    result = cv2.addWeighted(binary_image, 0.7, lines, 0.3, 0)
    result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)

    return result


def preprocess_image(image_input, display: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess an image for Sudoku grid detection.

    Steps: denoise, normalize, threshold, connect lines, optional rotation.

    Args:
        image_input: Either a file path (str) or numpy array (pre-loaded image)
        display: Whether to display intermediate steps

    Returns: (original, enhanced_grayscale, preprocessed_binary)
        - original: Original input image
        - enhanced_grayscale: Brightness-corrected, denoised grayscale (for straightening)
        - preprocessed_binary: Binary image for grid detection
    """
    if isinstance(image_input, str):
        print(f"\n{'='*60}")
        print(f"Processing: {image_input}")
        print('='*60)
        original = cv2.imread(image_input)
        if original is None:
            raise ValueError(f"Could not read {image_input}")
    else:
        print(f"\n{'='*60}")
        print(f"Processing: <image array>")
        print('='*60)
        original = image_input

    print("\n[1] Converting to grayscale")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    raw_noise = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise_type = detect_noise_type(gray)

    print(f"\n[2] Preprocessing strategy")
    print(f"  Raw noise level: {raw_noise:.1f}, Type: {noise_type}")

    if raw_noise > 15000 and noise_type == 'gaussian':
        print("  -> ULTRA-HIGH Gaussian noise: normalize → heavy denoise")

        print("\n[3] Brightness normalization")
        normalized = normalize_brightness(gray)

        print("\n[4] Heavy noise removal")
        denoised = remove_noise(normalized)
    else:
        print("  -> Standard: light denoise → normalize (best_one_yet approach)")

        print("\n[3] Light noise removal")
        denoised = remove_salt_pepper_noise(gray, raw_noise)

        print("\n[4] Brightness normalization")
        denoised = normalize_brightness(denoised)

    if np.mean(denoised) < 100 and np.sum(denoised < 128) / denoised.size > 0.6:
        print("\n[5] Inverting (white-on-black detected)")
        denoised = cv2.bitwise_not(denoised)

    print("\n[6] Smoothing")
    smooth = cv2.GaussianBlur(denoised, (5, 5), 0)

    if raw_noise <= 15000 or noise_type != 'gaussian':
        print("\n[7] Thresholding (simple adaptive - best for uneven lighting)")
        thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 3)
    else:
        print("\n[7] Thresholding (subimage-based - best for noisy images)")
        thresh = adaptive_threshold_subimages(smooth, grid_size=4)

    print("\n[8] Completing grid lines")
    completed = complete_grid_lines(thresh)

    print("\n[9] Checking rotation")
    rotation_angle = detect_rotation_robust(completed)

    if rotation_angle is not None:
        print(f"  -> Rotating by {rotation_angle:.2f}°")
        rotated = apply_rotation(denoised, rotation_angle)
        smooth = cv2.GaussianBlur(rotated, (5, 5), 0)

        if raw_noise <= 15000 or noise_type != 'gaussian':
            thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 3)
        else:
            thresh = adaptive_threshold_subimages(smooth, grid_size=4)

        final = complete_grid_lines(thresh)
        enhanced_gray = rotated
    else:
        print("  -> No rotation needed")
        final = completed
        enhanced_gray = denoised

    print("\n[10] Final cleanup")

    original_noise = cv2.Laplacian(gray, cv2.CV_64F).var()

    kernel = np.ones((2, 2), np.uint8)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel, iterations=1)

    print(f"\n{'='*60}")
    print("✓ Complete")
    print('='*60)

    if display:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title('Grayscale')
        axes[0].axis('off')

        axes[2].imshow(denoised, cmap='gray')
        axes[2].set_title('Denoised')
        axes[2].axis('off')

        axes[3].imshow(normalized, cmap='gray')
        axes[3].set_title('Normalized')
        axes[3].axis('off')

        axes[4].imshow(smooth, cmap='gray')
        axes[4].set_title('Smoothed')
        axes[4].axis('off')

        axes[5].imshow(thresh, cmap='gray')
        axes[5].set_title('Threshold')
        axes[5].axis('off')

        axes[6].imshow(completed, cmap='gray')
        axes[6].set_title('Lines Completed')
        axes[6].axis('off')

        axes[7].imshow(final, cmap='gray')
        axes[7].set_title('FINAL')
        axes[7].axis('off')

        plt.tight_layout()
        plt.show()

    return original, enhanced_gray, final


def resize_image(image, max_width=800):
    """Resize while maintaining aspect ratio."""
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return image
