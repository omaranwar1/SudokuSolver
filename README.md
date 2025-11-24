# Sudoku Solver - Computer Vision Project

**Course**: CSCE 4603 Fundamentals of Computer Vision
**Team Members**: Omar Anwar, Amal Fouda, Farida Bey, Ahmed Elbarbary

## Overview

This project implements a **Sudoku Solver** using Computer Vision techniques. The system captures a Sudoku puzzle from a real-world image, processes it through various image processing steps, and ultimately solves the puzzle.

### Real-World Inspiration

This project is inspired by the 2009 iPhone application **"Sudoku Grab"** by Kevin Gibbon.
## Project Milestones

### Milestone 1 (Due: 22/11/25) - 12 marks
- **Preprocessing of the captured image** (4 marks) ✅
- **Outer frame isolation** (4 marks) ✅
- **Outer frame corners identification** (2 marks) ✅
- **Grid straightening into a square** (2 marks) - Pending

## Current Implementation (Branch 2)

### ✅ Image Preprocessing Module (4 marks)

**Status**: Complete

This module implements robust image preprocessing for Sudoku grid detection, handling various image quality issues: noise, uneven lighting, poor contrast.

#### Features Implemented:

**1. Adaptive Noise Detection**
- Automatically identifies noise type (Gaussian vs salt-and-pepper)
- Uses Laplacian variance for noise severity measurement
- Analyzes extreme pixel values to classify noise

**2. Multi-Level Denoising**
- **Ultra-high Gaussian noise**: Heavy NLM denoising (h=40) + bilateral filtering
- **High Gaussian noise**: Strong NLM (h=30) + bilateral filtering
- **Moderate Gaussian noise**: Balanced NLM (h=20) + light bilateral
- **Salt-and-pepper noise**: Progressive median filtering (multiple passes)

**3. Brightness Normalization**
- Gamma correction for very dark/bright images (γ=0.6 to 3.5)
- Local brightness equalization using flat-field correction
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast
- Final normalization to 20-235 range

**4. Adaptive Thresholding**
- Simple adaptive thresholding for clean images
- Subimage-based adaptive thresholding for complex lighting (4×4 grid analysis)
- Histogram quality analysis to select optimal method per region

**5. Grid Line Completion**
- Conservative morphological operations
- Horizontal and vertical line extraction
- Small gap filling without adding noise

**6. Rotation Correction** (Optional)
- Detects rotation using Hough line transform
- Only applies if confident (>4° rotation, consistent measurements)
- Prevents unnecessary transformations

#### Technical Details:

**Algorithms Used**:
- Non-local means (NLM) denoising: Preserves fine details while removing noise
- Bilateral filtering: Maintains edges while smoothing
- Median filtering: Removes impulse noise effectively
- Morphological operations: Line completion and noise removal

**Pipeline**:
```
Input Image
    ↓
Grayscale Conversion
    ↓
Noise Detection (Gaussian/Salt-Pepper)
    ↓
Adaptive Denoising (based on noise type)
    ↓
Brightness Normalization (Gamma + CLAHE)
    ↓
Adaptive Thresholding
    ↓
Grid Line Completion
    ↓
(Optional) Rotation Correction
    ↓
Binary Preprocessed Image
```

#### References:
- Gonzalez & Woods, "Digital Image Processing" (4th Ed), Chapters 3, 9
- Buades et al. (2005): "A non-local algorithm for image denoising"
- Tomasi & Manduchi (1998): "Bilateral filtering for gray and color images"
- Bradski & Kaehler, "Learning OpenCV" (2008)

---

### ✅ Grid Detection Module (6 marks: 4 + 2)

**Status**: Complete

This module implements dual-method grid detection combining contour-based and line-based approaches for robust Sudoku grid isolation and corner identification.

#### Detection Methods:

**Method 1: Contour-Based Detection** (Optimal for clear grids)
- Uses Suzuki-Abe (1985) algorithm for contour extraction
- Douglas-Peucker polygon approximation for simplification
- Convex hull computation for corner identification
- Multi-pass approximation with varying epsilon values (0.01-0.05)
- Best for: Clear grid borders, good contrast images

**Method 2: Line-Based Hough Transform** (Robust for faint/occluded grids)
- Probabilistic Hough Transform for line detection
- Line clustering with IQR outlier removal
- Intersection-based corner computation
- Equation-based refinement for precision
- Best for: Faint grids, broken lines, occluded corners

#### Adaptive Selection Logic:

The system runs both methods in parallel and selects the best result using:

**1. Composite Scoring**:
- Quality score (0-1): Transform quality based on corner geometry
- Aspect ratio score: How close to square (1.0 = perfect)
- Coverage score: What % of image does grid occupy

**2. Selection Strategy**:
- **Contour Bias**: Requires line-based to score >0.075 higher to be selected
- **Blockage Detection**: Checks corner displacement between methods (>6% diagonal)
- **Coverage Override**: If corners blocked AND coverage gap <20%, use line-based

**3. Validation**:
- Aspect ratio validation (tolerance: ±20%)
- Corner geometry validation (angles 60-120°, convexity check)
- Quality scoring for transformation assessment

#### Corner Identification (2 marks):

**Precision Extraction**:
- Four-corner detection from contour or line intersections
- Ordering: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
- Geometry validation to ensure valid quadrilateral
- Quality scoring based on aspect ratios

**Algorithms**:
- Convex hull for contour-based corners
- Sklansky's algorithm for hull computation
- Line intersection computation for Hough-based corners
- Centroid-based angle sorting for proper ordering

#### Visualization Features:

- 9×9 grid overlay showing cell structure
- Corner markers with labels
- Detection method visualization
- Quality score display

#### Technical Algorithms:

1. **Suzuki-Abe (1985)**: Topological structural analysis
2. **Douglas-Peucker (1973)**: Polygon approximation
3. **Hough (1962)**: Line detection through parameter space voting
4. **Sklansky (1982)**: Convex hull computation

#### References:
- Suzuki & Abe (1985): "Topological Structural Analysis of Digitized Binary Images"
- Hough (1962): "Method and means for recognizing complex patterns"
- Douglas & Peucker (1973): "Algorithms for the reduction of points"
- Sklansky (1982): "Finding the convex hull of a simple polygon"

## Installation

### Requirements
- Python 3.7 or higher
- OpenCV 4.8.0+
- NumPy 1.24.0+
- SciPy (for preprocessing)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Current Functionality

```bash
# Test preprocessing on an image
python -c "from src.preprocessing import preprocess_image; preprocess_image('01.jpg', display=True)"

# Test grid detection
python -c "from src.grid_detection import find_largest_contour, find_grid_corners; import cv2; img = cv2.imread('01.jpg'); print('Testing grid detection...')"
```

## Project Structure

```
SudokuSolver/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Entry point for module execution
│   ├── preprocessing.py         # Image preprocessing (4 marks) ✅
│   └── grid_detection.py        # Grid detection + corners (6 marks) ✅
├── process_image.py             # Convenience script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Next Steps

Perspective transformation (2 marks)
- Homography computation
- Grid straightening to 450×450

Complete pipeline integration

## Development Notes

**Next Milestone**: Milestone 2 - December 4, 2025
