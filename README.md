# Sudoku Solver - Computer Vision Project

**Course**: CSCE 4603 Fundamentals of Computer Vision
**Team Members**: Omar Anwar, Amal Fouda, Ahmed Elbarbary, Farida Bey

## Table of Contents

- [Overview](#overview)
- [Project Milestones](#project-milestones)
- [Milestone 1 Implementation](#milestone-1-implementation)
- [Milestone 2 Implementation](#milestone-2-implementation)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Implementation](#technical-implementation)
- [Results](#results)
- [References](#references)

## Overview

This project implements a **Sudoku Solver** using Computer Vision techniques. The system captures a Sudoku puzzle from a real-world image (via camera), processes it through various image processing steps, and ultimately solves the puzzle.

The project demonstrates practical applications of:
- Robust image preprocessing
- Image enhancement and noise attenuation
- Morphological operations
- Hough transform for line detection
- Geometric transformations
- Pattern matching for OCR (Milestone 2)

## Project Milestones

- **Milestone 1**: Robust grid detection and perspective normalization to produce a clean 450x450 Sudoku grid (contour-based with Hough fallback).
- **Milestone 2**: OCR + solving pipeline with template-based digit recognition, conflict cleanup, and solution rendering on the straightened grid.

## Milestone 1 Implementation

### Processing Pipeline

The Milestone 1 pipeline consists of five main stages:

```
┌─────────────────┐
│  Input Image    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1. Preprocessing│ - Grayscale conversion
│                 │ - Gaussian blur (noise reduction)
│                 │ - Adaptive thresholding
│                 │ - Morphological operations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Contour      │ - Find all contours
│    Detection    │ - Filter by area
│                 │ - Select largest quadrilateral
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Corner       │ - Convex hull computation
│    Detection    │ - Extract 4 corners
│                 │ - Order corners (TL, TR, BR, BL)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Perspective  │ - Compute homography matrix
│    Transform    │ - Warp to square grid
│                 │ - Output: 450x450 pixels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Straightened    │
│ Sudoku Grid     │
└─────────────────┘
```

During processing, the solver runs both contour-based detection (preferred) and a line-based Hough variant; it keeps the contour corners when available, falls back to line-based corners if contour detection fails, and may reuse the line-based corners later if OCR shows conflicts.

## Milestone 2 Implementation

Milestone 2 extends the pipeline to run OCR, resolve conflicts, and output a solved Sudoku.

### Extended Pipeline

```
┌────────────────────────┐
│ Straightened Binary    │
│ Grid (450x450)         │
└──────────┬─────────────┘
           │ reinforce + clean lines
           ▼
┌────────────────────────┐
│ Cell Extraction        │ - 9x9 split
│ (50x50 crops)          │ - border clearing + save crops
└──────────┬─────────────┘
           │
           ▼
┌────────────────────────┐
│ Template OCR           │ - multi-font templates + stroke aug
│ (IoU + hole penalty)   │ - shift/scale search per cell
└──────────┬─────────────┘
           │ conflict cleanup & fallback corners
           ▼
┌────────────────────────┐
│ Sudoku Solver          │ - validate givens
│ + Solution Overlay     │ - backtracking with step cap
└────────────────────────┘
```

Corner detection runs contour-first (largest contour + corner approximation) and falls back to the Hough line-based corners when contours fail or when OCR needs a cleaner set of corners.

### Digit Template Generation

**Purpose**: Provide stable reference digits for template matching.
**Details**:
- `python -m src.generate_digit_templates` renders 50x50 PNGs for digits 1-9 into `digit_templates/` using bundled bold fonts (DejaVu Sans, Albert Sans, Noto Sans JP, Lato).
- Each template is centered and optionally augmented with eroded/dilated strokes to absorb stroke width changes during scanning.
- OCR fails fast with a clear error if any digit templates are missing.

### OCR Pipeline

**Purpose**: Convert the straightened binary grid into a reliable 9x9 numeric board.
**Steps**:
- Reinforce weak/missing lines by scanning the 10 expected grid positions in each direction, then strip grid lines before OCR.
- Split into 9x9 cells (50x50 crops), clear borders, and reject empties using coverage + component heuristics; recenters the main blob when safe.
- Score candidates against multi-font templates using overlap/IoU with a hole-count penalty; searches small shift/scale/polarity variations per cell.
- Resolve duplicates with `resolve_conflicts`; if contour-based OCR shows conflicts, reruns OCR on line-detected corners and keeps the board with fewer conflicts/more givens.
- Optional debug crops saved to `output/{image}_cells/` (and `{image}_cells_line/` when the fallback path runs).

### Sudoku Solving & Rendering

**Purpose**: Deliver the solved puzzle alongside visual feedback.
**Details**:
- Backtracking solver in `src/solver.py` validates givens and caps search steps (default 200k) before attempting a solution.
- Successful solves render digits back onto the straightened image as `{image}_solved_overlay.jpg` and print the board plus any cleaned conflicts to the console.
- CLI options: `--no-save` to skip intermediate exports, `--binary-grid` to feed an already straightened binary grid directly into OCR/solving.

## Installation

### Requirements

- Python 3.7 or higher
- OpenCV 4.8.0+
- NumPy 1.24.0+
- Matplotlib 3.7.0+

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SudokuSolver
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install opencv-python numpy matplotlib
   ```

## Usage

### Basic Usage

Process a single image (runs OCR + solver and saves intermediates):

```bash
python process_image.py --image 01.jpg
```

Or call the module directly:

```bash
python -m src.sudoku_solver --image 01.jpg
```

Optional flags:

```bash
python -m src.sudoku_solver --image 01.jpg --size 600      # change straightened grid size (default 450)
python -m src.sudoku_solver --image 01.jpg --no-save       # skip saving intermediate images/cell crops
python -m src.sudoku_solver --binary-grid path/to/grid.png # OCR-only on a pre-straightened binary grid (prints board, no solve/overlay)
```

### Output Files

When saving is enabled (default), the application writes the following to the `output/` directory:

- `{image}_original.jpg` - Resized original image
- `{image}_preprocessed.jpg` - Binary preprocessed image
- `{image}_contour_detection.jpg` - Detected contour and corners
- `{image}_enhanced_gray.jpg` - Brightness-normalized grayscale used for straightening
- `{image}_straightened.jpg` - Final straightened color grid
- `{image}_straightened_binary.jpg` - Binary straightened grid (for OCR)
- `{image}_grid_9x9_visualization.jpg` - Grid lines drawn on the detected corners
- `{image}_straightened_binary_reinforced.jpg` - Binary grid with missing lines filled (when needed)
- `{image}_solved_overlay.jpg` - Solution rendered on the straightened grid (when solvable)
- `{image}_cells/` (and `{image}_cells_line/` when fallback is used) - 50x50 crops for each cell used by the OCR
- Line-based fallback (only when invoked): `{image}_straightened_line.jpg`, `{image}_straightened_binary_line.jpg`, `{image}_straightened_binary_line_reinforced.jpg`

## Technical Implementation

### 1. Image Preprocessing

**Purpose**: Prepare the image for reliable grid detection by reducing noise, normalizing brightness, and connecting grid lines.

**Techniques**:

- **Grayscale Conversion**: Reduces 3-channel RGB to single intensity channel
  - Formula: `Gray = 0.299*R + 0.587*G + 0.114*B`

- **Noise + Brightness Handling**: Estimate noise (Laplacian variance) and type (Gaussian vs salt/pepper) to pick a denoise path (fastNlMeans + bilateral for Gaussian; median/bilateral for salt/pepper). Apply brightness normalization/gamma and optional inversion when the input is white-on-black.

- **Smoothing + Thresholding**: Gaussian blur (5×5) followed by either standard adaptive thresholding (15×15, C=3, inverted) or a subimage-based adaptive path on very noisy inputs.

- **Grid Completion**: Horizontal/vertical morphology (open + dilate) blended back into the binary to reconnect broken grid lines, then light open/close cleanup.

- **Rotation Check**: Detects coarse rotation; when rotation is detected, reprocesses (denoise → normalize → threshold → grid completion) on the rotated view.

**References**:
- Gonzalez & Woods, "Digital Image Processing" (4th Edition), Chapters 3, 9
- Bradski & Kaehler, "Learning OpenCV" (2008)

### 2. Outer Frame Isolation

**Purpose**: Locate the Sudoku grid boundary in the image.

**Algorithm**:

1. **Contour Detection** using Suzuki-Abe algorithm (1985)
   - Mode: `RETR_EXTERNAL` - retrieves only outermost contours
   - Method: `CHAIN_APPROX_SIMPLE` - compresses contour representation

2. **Filtering**:
   - Sort contours by area (descending)
   - Filter by minimum area (≥20% of image area)
   - Approximate contour to polygon using Douglas-Peucker algorithm

3. **Validation**:
   - Accept contours with at least 4 vertices (accounts for perspective distortion)
   - Select the first area-qualified contour as the Sudoku grid

**References**:
- Suzuki & Abe, "Topological Structural Analysis of Digitized Binary Images" (1985)
- Douglas & Peucker, "Algorithms for the reduction of the number of points required to represent a digitized line" (1973)

### 3. Corner Detection

**Purpose**: Identify the four corner points of the grid for perspective transformation.

**Algorithm**:

1. **Convex Hull Computation**
   - Finds smallest convex polygon containing the contour
   - Uses Sklansky's algorithm (1982)

2. **Quadrilateral Approximation**
   - Multi-pass approximation on the convex hull; if 5-6 points appear, merge closest pairs; otherwise fall back to extreme-point selection

3. **Corner Ordering**
   - Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
   - Method: Sort by angle around the contour centroid, then normalize to a consistent TL→TR→BR→BL order
   - Essential for correct perspective transformation

### 4. Perspective Transformation

**Purpose**: Transform the trapezoidal grid into a perfect square for uniform processing.

**Mathematical Foundation**:

The transformation uses a **homography matrix** (3×3) to map points from source to destination:

```
[x']   [h11 h12 h13]   [x]
[y'] = [h21 h22 h23] × [y]
[w']   [h31 h32 h33]   [1]

where (x'/w', y'/w') is the transformed point
```

**Algorithm**:

1. **Define Correspondence**:
   - Source: 4 detected corners (distorted)
   - Destination: 4 corners of square (0,0), (450,0), (450,450), (0,450)

2. **Compute Homography**:
   - Uses Direct Linear Transformation (DLT)
   - Solves system of linear equations for 8 unknowns (h33=1)
   - Requires minimum 4 point correspondences

3. **Apply Warping**:
   - Bilinear interpolation for pixel values
   - Output: 450×450 pixel square grid

**Quality Assessment**:
- Calculate aspect ratios of detected quadrilateral
- Score range: 0.0 (poor) to 1.0 (perfect square)
- Quality score is logged to guide debugging (higher is closer to square)

**References**:
- Hartley & Zisserman, "Multiple View Geometry in Computer Vision" (2004), Chapter 4
- Szeliski, "Computer Vision: Algorithms and Applications" (2010)
- OpenCV Documentation: [Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)

### Alternative Method: Line-Based Grid Detection (Hough Transform)

The detector also runs a line-based path using probabilistic Hough to recover corners when contours are weak.

**Purpose**: Detect straight lines to infer grid boundaries.

**Approach**:
- Probabilistic Hough (two passes: threshold≈100, then 50 if weak) with `minLineLength` ≈ 0.3×min(image dimension) in the first pass (≈0.2× in the fallback) and `maxLineGap` 20–30.
- Classify horizontal/vertical lines, cluster into up to 10 per direction, and take outermost clusters as boundaries.
- Optional slope-consistency check refines corners from averaged line equations when the aspect ratio stays near-square.

**References**:
- Hough, P.V.C., "Method and means for recognizing complex patterns" (1962)
- Duda & Hart, "Use of the Hough transformation to detect lines and curves in pictures" (1972)

### 5. Digit Extraction & OCR (Milestone 2)

**Purpose**: Convert the straightened binary grid into a reliable 9x9 board of digits.

**Algorithm**:
- **Grid conditioning**: Fill missing lines via expected-position scan, then remove grid lines using directional morphology to isolate digits.
- **Cell processing**: Split into 9x9 cells (50×50), clear borders, reject empty cells with coverage/component checks, and recenter the main blob when safe.
- **Template matching**: Compare against multi-font templates (with dilated/eroded variants) using overlap/IoU plus a hole-count penalty to separate shapes like 6/8/9. Searches small shift/scale/polarity variants per cell to absorb misalignment.
- **Conflict cleanup**: `resolve_conflicts` drops duplicates/low-confidence hits; if contour-based OCR shows conflicts, OCR reruns with line-detected corners and keeps the cleaner board.

### 6. Sudoku Solving & Visualization (Milestone 2)

**Purpose**: Solve the detected puzzle and present the answer visually.

**Algorithm**:
- Validate givens (rows/cols/blocks) before solving to short-circuit impossible puzzles.
- Backtracking solver (`solve_puzzle`) with a configurable step cap (default: 200k expansions) to avoid runaway searches.
- Renders the solved digits onto the straightened grid (`{image}_solved_overlay.jpg`) while keeping the detected givens untouched; cell crops and overlays are saved when intermediate exports are enabled.

## Results

### Performance

- **Typical processing time**: 30 to 40 seconds per image on the provided samples (hardware-dependent)

### Output Examples

After processing, each image generates:
1. **Preprocessed binary image** - Shows cleaned, thresholded grid
2. **Contour detection** - Visualizes detected boundary and corners
3. **Straightened grid** - Final 450×450 normalized output
4. **Solved overlay** - Solution digits rendered on the straightened grid (when solvable)

## References

### Academic Papers

1. **Suzuki, S., & Abe, K.** (1985). "Topological Structural Analysis of Digitized Binary Images by Border Following". *Computer Vision, Graphics, and Image Processing*, 30(1), 32-46.

2. **Hough, P.V.C.** (1962). "Method and means for recognizing complex patterns". U.S. Patent 3,069,654.

3. **Duda, R.O., & Hart, P.E.** (1972). "Use of the Hough transformation to detect lines and curves in pictures". *Communications of the ACM*, 15(1), 11-15.

4. **Douglas, D.H., & Peucker, T.K.** (1973). "Algorithms for the reduction of the number of points required to represent a digitized line or its caricature". *Cartographica*, 10(2), 112-122.

5. **Sklansky, J.** (1982). "Finding the convex hull of a simple polygon". *Pattern Recognition Letters*, 1(2), 79-83.

### Textbooks

1. **Gonzalez, R.C., & Woods, R.E.** (2018). *Digital Image Processing* (4th Edition). Pearson.
   - Chapter 3: Intensity Transformations and Spatial Filtering
   - Chapter 9: Morphological Image Processing
   - Chapter 10: Image Segmentation

2. **Bradski, G., & Kaehler, A.** (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media.
   - Image preprocessing techniques
   - Contour detection and analysis

3. **Hartley, R., & Zisserman, A.** (2004). *Multiple View Geometry in Computer Vision* (2nd Edition). Cambridge University Press.
   - Chapter 4: Estimation - 2D Projective Transformations

4. **Szeliski, R.** (2010). *Computer Vision: Algorithms and Applications*. Springer.
   - Image transformations and warping

### Online Resources

1. **OpenCV Documentation**:
   - [Image Processing](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)
   - [Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)
   - [Contour Features](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
   - [Hough Line Transform](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)

2. **Wikipedia**:
   - [Homography](https://en.wikipedia.org/wiki/Homography_(computer_vision))
   - [Perspective Transform](https://en.wikipedia.org/wiki/Perspective_transform)
   - [Adaptive Thresholding](https://en.wikipedia.org/wiki/Thresholding_(image_processing))
