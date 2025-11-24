# Sudoku Solver - Computer Vision Project

**Course**: CSCE 4603 Fundamentals of Computer Vision
**Project Weight**: 30%
**Team Members**: Omar Anwar

## Table of Contents

- [Overview](#overview)
- [Project Milestones](#project-milestones)
- [Milestone 1 Implementation](#milestone-1-implementation)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Implementation](#technical-implementation)
- [Results](#results)
- [References](#references)
- [Future Work](#future-work)

## Overview

This project implements a **Sudoku Solver** using Computer Vision techniques. The system captures a Sudoku puzzle from a real-world image (via camera), processes it through various image processing steps, and ultimately solves the puzzle.

The project demonstrates practical applications of:
- Robust image preprocessing
- Image enhancement and noise attenuation
- Morphological operations
- Hough transform for line detection
- Geometric transformations
- Pattern matching for OCR (Milestone 2)

### Real-World Inspiration

This project is inspired by the 2009 iPhone application **"Sudoku Grab"** by Kevin Gibbon. For more details on the original implementation, see this [blog post](https://github.com/kevingibbon/SudokuGrab).

## Project Milestones

### Milestone 1 (Due: 22/11/25) - 12 marks ✅

- ✅ **Preprocessing of the captured image** (4 marks)
- ✅ **Outer frame isolation** (4 marks)
- ✅ **Outer frame corners identification** (2 marks)
- ✅ **Grid straightening into a square** (2 marks)

### Milestone 2 (Due: 4/12/25) - 18 marks

- ⏳ **Basic OCR with pattern matching** (8 marks)
- ⏳ **Solving the extracted sudoku puzzle** (2 marks)
- ⏳ **Documentation, demo, and discussion** (8 marks)

## Milestone 1 Implementation

### Architecture

```
SudokuSolver/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── preprocessing.py          # Image preprocessing module
│   ├── grid_detection.py         # Grid and corner detection
│   ├── perspective_transform.py  # Perspective transformation
│   ├── sudoku_solver.py          # Main application pipeline
│   └── __main__.py              # Entry point for module execution
├── process_image.py              # Convenience script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

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

Process a single image:

```bash
python process_image.py --image 01.jpg
```

Or using the module:

```bash
python -m src.sudoku_solver --image 01.jpg
```

### Advanced Options

```bash
# Specify output directory
python process_image.py --image 01.jpg --output my_results/

# Custom output grid size
python process_image.py --image 01.jpg --size 600

# Don't save intermediate images
python process_image.py --image 01.jpg --no-save

# Show help
python process_image.py --help
```

### Output Files

The application generates the following outputs in the `output/` directory:

- `{image}_original.jpg` - Resized original image
- `{image}_preprocessed.jpg` - Binary preprocessed image
- `{image}_contour_detection.jpg` - Detected contour and corners
- `{image}_straightened.jpg` - Final straightened color grid
- `{image}_straightened_binary.jpg` - Binary straightened grid (for OCR)

### Example

```bash
$ python process_image.py --image 01.jpg

============================================================
Processing: 01.jpg
============================================================

[1/5] Loading image...
      Image size: 800x800

[2/5] Preprocessing image...
      - Converting to grayscale
      - Applying Gaussian blur
      - Adaptive thresholding
      - Morphological operations

[3/5] Detecting grid contour...
      Grid contour found (area: 545827 pixels)

[4/5] Identifying corners...
      Corners detected:
        Top-Left:     (38.0, 45.0)
        Top-Right:    (764.0, 18.0)
        Bottom-Right: (778.0, 754.0)
        Bottom-Left:  (15.0, 774.0)
      Quality score: 0.96

[5/5] Applying perspective transformation...
      Grid straightened to 450x450

============================================================
Processing complete!
Results saved to: output/
============================================================
```

## Technical Implementation

### 1. Image Preprocessing

**Purpose**: Prepare the image for reliable grid detection by reducing noise and enhancing features.

**Techniques**:

- **Grayscale Conversion**: Reduces 3-channel RGB to single intensity channel
  - Formula: `Gray = 0.299*R + 0.587*G + 0.114*B`

- **Gaussian Blur**: Reduces noise while preserving edges
  - Kernel size: 9×9
  - Removes high-frequency noise from camera sensor

- **Adaptive Thresholding**: Creates binary image robust to lighting variations
  - Method: `ADAPTIVE_THRESH_GAUSSIAN_C`
  - Block size: 11×11 pixels
  - Constant: C=2
  - Inverted: Grid lines become white (foreground)

- **Morphological Closing**: Fills small gaps in grid lines
  - Operation: Dilation followed by erosion
  - Kernel: 3×3 rectangular structuring element
  - Iterations: 1

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
   - Accept contours with 4-8 vertices (accounts for perspective distortion)
   - Select largest valid contour as the Sudoku grid

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
   - If exactly 4 points: use directly
   - Otherwise: find 4 extreme points using coordinate analysis

3. **Corner Ordering**
   - Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
   - Method: Calculate angles from center point
   - Essential for correct perspective transformation

**Corner Selection Logic**:
```python
# Find extreme corners based on coordinate sums/differences
Top-Left:     min(x + y)  # Closest to origin
Top-Right:    min(y - x)  # Top edge, rightmost
Bottom-Right: max(x + y)  # Farthest from origin
Bottom-Left:  min(x - y)  # Left edge, bottommost
```

**References**:
- Sklansky, "Finding the convex hull of a simple polygon" (1982)
- Standard computer vision practices for perspective transforms

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
- Typical acceptable threshold: ≥0.70

**References**:
- Hartley & Zisserman, "Multiple View Geometry in Computer Vision" (2004), Chapter 4
- Szeliski, "Computer Vision: Algorithms and Applications" (2010)
- OpenCV Documentation: [Geometric Transformations](https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html)

### Alternative Method: Hough Transform

While the primary implementation uses contour detection, the codebase also includes **Hough Line Transform** as an alternative approach:

**Purpose**: Detect straight lines in the image to find grid structure.

**Parameters**:
- ρ (rho): 1 pixel resolution
- θ (theta): 1 degree resolution (π/180)
- Threshold: 200 votes minimum

**References**:
- Hough, P.V.C., "Method and means for recognizing complex patterns" (1962)
- Duda & Hart, "Use of the Hough transformation to detect lines and curves in pictures" (1972)

## Results

### Test Cases

The implementation was tested on 16 sample images with varying conditions:
- Different lighting conditions
- Various camera angles
- Different grid styles
- Multiple difficulty levels

### Sample Results

| Image | Grid Detection | Corner Detection | Quality Score | Status |
|-------|----------------|------------------|---------------|--------|
| 01.jpg | ✅ | ✅ | 0.96 | Success |
| 05.jpg | ✅ | ✅ | 0.93 | Success |
| 10.jpg | ✅ | ✅ | 0.96 | Success |

### Performance

- **Average processing time**: ~0.5-1 second per image
- **Success rate**: High (tested on 16 varied images)
- **Quality scores**: Typically 0.90-0.98 for well-captured images

### Output Examples

After processing, each image generates:
1. **Preprocessed binary image** - Shows cleaned, thresholded grid
2. **Contour detection** - Visualizes detected boundary and corners
3. **Straightened grid** - Final 450×450 normalized output

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

2. **Sudoku Grab** (Original Inspiration):
   - [Kevin Gibbon's Blog Post](https://github.com/kevingibbon/SudokuGrab)
   - [App Store Listing](https://apps.apple.com/us/app/sudoku-grab/id303705581)

3. **Wikipedia**:
   - [Homography](https://en.wikipedia.org/wiki/Homography_(computer_vision))
   - [Perspective Transform](https://en.wikipedia.org/wiki/Perspective_transform)
   - [Adaptive Thresholding](https://en.wikipedia.org/wiki/Thresholding_(image_processing))

## Future Work

### Milestone 2 Tasks

1. **Optical Character Recognition (OCR)**:
   - Implement pattern matching for digit recognition (1-9)
   - Extract digits from each cell of the straightened grid
   - Handle empty cells correctly

2. **Sudoku Solving Algorithm**:
   - Integrate existing solving algorithm (backtracking or constraint propagation)
   - Validate extracted puzzle before solving
   - Handle invalid/unsolvable puzzles gracefully

3. **Enhanced Documentation**:
   - Create video demonstration
   - Document test cases and results
   - Prepare for project discussion

### Potential Improvements

- Real-time processing via webcam
- Mobile application development
- Machine learning for improved OCR accuracy
- Support for different Sudoku variants (16×16, Samurai, etc.)
- GUI for easier user interaction
- Batch processing for multiple images

## Project Structure

```
SudokuSolver/
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── preprocessing.py          # Image preprocessing (Milestone 1)
│   ├── grid_detection.py         # Grid and corner detection (Milestone 1)
│   ├── perspective_transform.py  # Geometric transformation (Milestone 1)
│   └── sudoku_solver.py          # Main pipeline (Milestone 1)
│
├── 01.jpg - 16.jpg               # Test images
├── output/                       # Generated results (gitignored)
├── process_image.py              # Convenience script
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── Project Specification.pdf     # Course project specification
└── README.md                     # This documentation
```

## Development Notes

### Git Practices

Following best practices for academic collaboration:

- ✅ **Distributed commits**: Each feature in separate commit
- ✅ **Granular changes**: Small, focused commits
- ✅ **Self-explanatory messages**: Clear descriptions with context
- ✅ **Comprehensive documentation**: Inline comments and references

### Commit History

Branch Structure (Omar Anwar):
```
omar_anwar1/01-setup-and-preprocessing (starting from eb82a5c)
└── Add comprehensive image preprocessing module
    • Adaptive noise detection & removal (Gaussian/salt-pepper)
    • Brightness normalization (gamma + CLAHE)
    • Adaptive thresholding
    • Grid line completion
    • Branch 1 README: Documents preprocessing (4/12 marks)

omar_anwar1/02-grid-detection (based on 01)
└── Implement dual-method grid detection and corner identification
    • Contour-based detection (Suzuki-Abe)
    • Line-based Hough Transform detection
    • Adaptive selection with quality scoring
    • Corner extraction & validation
    • Branch 2 README: Documents preprocessing + grid detection (10/12 marks)

omar_anwar1/03-perspective-transform (based on 02)
└── Implement perspective transformation for grid straightening
    • Homography matrix computation (DLT)
    • Perspective warp to 450×450 square
    • Quality assessment
    • Branch 3 README: Documents all core components (12/12 marks)

omar_anwar1/04-complete-pipeline (based on 03)
├── Integrate complete Milestone 1 pipeline
│   • Main application orchestration
│   • Command-line interface
│   • Comprehensive logging & output
│
└── Complete comprehensive documentation
    • Full README with all technical details
    • Academic references throughout
    • Usage examples and results
    • Branch 4 README: Complete Milestone 1 documentation
```

Each branch builds incrementally on the previous one, with README documentation
growing from basic (Branch 1) to comprehensive (Branch 4).

## License

This project is submitted as coursework for CSCE 4603 Fundamentals of Computer Vision.

---

**Last Updated**: November 24, 2025
**Milestone 1 Status**: ✅ Complete
**Next Deadline**: Milestone 2 - December 4, 2025
