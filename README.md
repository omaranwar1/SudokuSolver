# ✨ Sudoku Solver — Computer Vision Project

<<<<<<< HEAD
**Course:** CSCE 4603 — Fundamentals of Computer Vision
**Project Weight:** 30%
**Team Member:** **Omar Anwar** **Farida Bey** **Amal Fouda** **Ahmed El Barbary**
=======
**Course**: CSCE 4603 Fundamentals of Computer Vision
**Team Members**: Omar Anwar, Amal Fouda, Ahmed barbary, Farida bey
>>>>>>> dd1f6d0 (fixed errors)

---

## 📚 Table of Contents

* [Overview](#overview)
* [Project Milestones](#project-milestones)
* [Milestone 1 Implementation](#milestone-1-implementation)
* [Installation](#installation)
* [Usage](#usage)
* [Technical Implementation](#technical-implementation)
* [Results](#results)
* [References](#references)
* [Future Work](#future-work)
* [Project Structure](#project-structure)
* [Development Notes](#development-notes)

---

## 🧩 Overview

<<<<<<< HEAD
This project implements a complete **Sudoku Solver** using fundamental CV techniques.
The system extracts a Sudoku puzzle from a real-world image, processes it through a structured pipeline, detects the grid, corrects perspective distortions, and prepares it for digit extraction and solving.

This work highlights practical mastery of:

* Robust preprocessing
* Noise reduction & image enhancement
* Morphological ops
* Hough transform
* Perspective correction
* Pattern matching OCR (Milestone 2)

### 🌟 Inspiration

Inspired by the 2009 iPhone app **Sudoku Grab** by Kevin Gibbon.
Ref: [Sudoku Grab Blog](https://github.com/kevingibbon/SudokuGrab)

---

## 🚀 Project Milestones

### **Milestone 1 — Preprocessing & Grid Extraction** (12 marks) ✔️

* Preprocessing (4/4)
* Outer frame isolation (4/4)
* Corner detection (2/2)
* Perspective straightening (2/2)

➡️ **Status: 100% Complete**

### **Milestone 2 — OCR & Solving** (18 marks)

* OCR via pattern matching (0/8)
* Sudoku solver (0/2)
* Documentation & demo (0/8)

➡️ **Status: In Progress**

---

## 🛠️ Milestone 1 Implementation

### 📦 Folder Architecture

```
SudokuSolver/
├── src/
│   ├── preprocessing.py         
│   ├── grid_detection.py        
│   ├── perspective_transform.py 
│   ├── sudoku_solver.py         
│   └── __main__.py              
├── process_image.py             
├── requirements.txt             
└── README.md                    
```
=======
>>>>>>> dd1f6d0 (fixed errors)

---

## 🔧 Processing Pipeline (Milestone 1)

```
      ┌─────────────────────┐
      │     Input Image     │
      └──────────┬──────────┘
                 ▼
┌─────────────────────────────────┐
│ 1. Preprocessing                │
│    - Grayscale                  │
│    - Gaussian Blur              │
│    - Adaptive Thresholding      │
│    - Morphology                │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│ 2. Contour Detection            │
│    - Find contours              │
│    - Filter by area             │
│    - Select grid polygon        │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│ 3. Corner Detection             │
│    - Convex hull                │
│    - Extract 4 corners          │
│    - Order corners              │
└──────────┬──────────────────────┘
           ▼
┌─────────────────────────────────┐
│ 4. Perspective Transform        │
│    - Compute homography         │
│    - Warp to 450×450 square     │
└─────────────────────────────────┘
```

---

## 💻 Installation

```
git clone <repo>
cd SudokuSolver
pip install -r requirements.txt
```

Or manually:

```
pip install opencv-python numpy matplotlib
```

---

## ▶️ Usage

### Basic Run

```
python process_image.py --image 01.jpg
```

<<<<<<< HEAD
### Advanced Options

```
python process_image.py --image 01.jpg --output results/
python process_image.py --image 01.jpg --size 600
python process_image.py --image 01.jpg --no-save
=======
Or using the module:

```bash
python -m src.sudoku_solver --image 01.jpg
```
>>>>>>> dd1f6d0 (fixed errors)
```

### Output (Generated Automatically)

<<<<<<< HEAD
* `*_original.jpg`
* `*_preprocessed.jpg`
* `*_contour_detection.jpg`
* `*_straightened.jpg`
* `*_straightened_binary.jpg`

---

## 🧠 Technical Implementation

### ⭐ 1. Preprocessing

* Grayscale
* Gaussian blur (9×9)
* Adaptive Gaussian threshold
* Morphological closing

> Based on Gonzalez & Woods + OpenCV best practices.

---

### ⭐ 2. Outer Frame Isolation

* Suzuki-Abe contour tracking
* Area filtering
* Douglas-Peucker polygon approximation

---

### ⭐ 3. Corner Detection

* Convex hull
* Extract 4 extreme points
* Sort into TL / TR / BR / BL

---

### ⭐ 4. Perspective Transform

* Compute homography
* Warp into normalized square
* Output: **450×450 px**

> Based on Hartley & Zisserman, Szeliski.

---

## 📈 Results

| Image  | Grid | Corners | Score | Status  |
| ------ | ---- | ------- | ----- | ------- |
| 01.jpg | ✔️   | ✔️      | 0.96  | Success |
| 05.jpg | ✔️   | ✔️      | 0.93  | Success |
| 10.jpg | ✔️   | ✔️      | 0.96  | Success |

**Average runtime:** ~0.5–1 sec
**Success rate:** High across 16 test images

---

## 📚 References

Academic papers and textbooks from:

* Suzuki & Abe
* Douglas & Peucker
* Sklansky
* Hough / Duda & Hart
* Gonzalez & Woods
* Hartley & Zisserman
* Szeliski

Plus:

* OpenCV documentation
* Sudoku Grab project

---

## 🔮 Future Work (Milestone 2)

### Core Tasks

* OCR using template / pattern matching
* Sudoku solving (backtracking / constraint propagation)
* Full documentation + video demo

### Possible Enhancements

* Real-time webcam mode
* ML-based OCR
* Support for exotic Sudoku variants
* GUI interface

---

## 📦 Project Structure

```
SudokuSolver/
├── src/
│   ├── preprocessing.py
│   ├── grid_detection.py
│   ├── perspective_transform.py
│   ├── sudoku_solver.py
│   └── __main__.py
├── 01.jpg–16.jpg
├── output/
├── process_image.py
├── requirements.txt
└── README.md
```

---

## 📝 Development Notes

### Git Workflow Highlights

* Each feature in its own branch
* Clean commit history
* Detailed README evolution
* Clear incremental improvements

### Branch Summary

* `01-setup-and-preprocessing`
* `02-grid-detection`
* `03-perspective-transform`
* `04-complete-pipeline`

Each branch expands documentation and functionality.

---

## 📌 License

Academic submission for CSCE 4603 — Fundamentals of Computer Vision.

---

**Last Updated:** Nov 24, 2025
**Milestone 1:** ✔️ Completed
**Next Deadline:** Milestone 2 — Dec 4, 2025
=======
The application generates the following outputs in the `output/` directory:

- `{image}_original.jpg` - Resized original image
- `{image}_preprocessed.jpg` - Binary preprocessed image
- `{image}_contour_detection.jpg` - Detected contour and corners
- `{image}_straightened.jpg` - Final straightened color grid
- `{image}_straightened_binary.jpg` - Binary straightened grid (for OCR)

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


>>>>>>> dd1f6d0 (fixed errors)
