# **Sudoku Solver - Computer Vision Project**

**CSCE 4603 Fundamentals of Computer Vision**  
**Omar Anwar, Amal Fouda, Ahmed Elbarbary, Farida Bey**

---

## **1. Introduction**

This project implements a complete **Sudoku Solver** using computer vision techniques. The system takes an image of a Sudoku puzzle, preprocesses it, detects the grid, extracts digits via OCR, solves the puzzle, and outputs the solution overlaid on the straightened grid. The implementation avoids machine learning for OCR, relying instead on **pattern matching and geometric transformations**.

---

## **2. Project Structure**

```
src/
├── __init__.py              # Package metadata
├── __main__.py              # Entry point
├── preprocessing.py         # Image enhancement and binarization
├── grid_detection.py        # Grid contour and corner detection
├── perspective_transform.py # Homography and grid straightening
├── ocr.py                   # Digit recognition via template matching
├── solver.py                # Backtracking Sudoku solver
├── sudoku_solver.py         # Main pipeline orchestrator
└── requirements.txt         # Dependencies
```

---

## **3. Pipeline Overview**

### **Step 1: Image Preprocessing (`preprocessing.py`)**

- **Input**: RGB image (file path or array).
- **Steps**:
  1. Convert to grayscale.
  2. Detect noise type (Gaussian/salt-and-pepper) and apply adaptive denoising.
  3. Normalize brightness using gamma correction and CLAHE.
  4. Apply adaptive thresholding (Otsu for good regions, Gaussian adaptive for poor ones).
  5. Complete broken grid lines using morphological operations.
  6. Detect and correct rotation via Hough line analysis.
- **Output**: Clean binary image with enhanced grid structure.

---

### **Step 2: Grid Detection (`grid_detection.py`)**

- **Hybrid Detection Strategy**:
  - **Contour-based**: Find largest quadrilateral contour → refine corners.
  - **Line-based**: Use Hough transform to detect grid lines → compute intersections.
  - **Adaptive Selection**: Score both methods based on quality, aspect ratio, and coverage → pick best.
- **Corner Refinement**:
  - Merge close points, refine via Harris corner detection, validate geometry.
- **Output**: Four ordered corners (TL, TR, BR, BL).

---

### **Step 3: Perspective Transformation (`perspective_transform.py`)**

- Compute homography matrix using `cv2.getPerspectiveTransform`.
- Map detected corners to a square (default 450×450).
- Apply warp transformation with bilinear interpolation.
- **Quality Check**: Evaluate aspect ratio and corner angles to reject poor transforms.

---

### **Step 4: Grid Reinforcement**

- After straightening, check for missing/broken grid lines.
- **Method**: Scan expected line positions, fill gaps if coverage < 30%.
- Ensures a complete 9×9 grid for reliable cell segmentation.

---

### **Step 5: OCR – Digit Recognition (`ocr.py`)**

- **Template Generation**: Create multiple variants of digits 1–9 with different fonts, thicknesses, rotations, and noise.
- **Cell Processing**:
  - Split grid into 81 cells.
  - Remove grid lines via morphology.
  - Detect largest connected component (the digit).
  - Normalize to 28×28 pixels.
- **Matching**:
  - Compare cell against all templates using normalized cross-correlation (NCC) and IoU.
  - Apply hole-count penalties (e.g., ‘8’ has 2 holes, ‘6’ has 1).
  - Accept digit if score > threshold and separation from second-best is sufficient.
- **Conflict Resolution**: Remove duplicate digits in rows/columns/blocks, keeping highest-confidence detection.

---

### **Step 6: Sudoku Solving (`solver.py`)**

- **Validation**: Check for duplicate givens in rows, columns, blocks.
- **Backtracking**: Recursive DFS with constraint propagation.
- **Step Limit**: Cap at 200,000 expansions to avoid infinite loops.
- **Output**: Solved 9×9 board or error message.

---

### **Step 7: Visualization & Output (`sudoku_solver.py`)**

- Overlay solution digits (green) onto straightened image.
- Save intermediate images for debugging.
- Print detected and solved boards in readable format.

---

## **4. Key Algorithms & Techniques**

| Module                | Technique                                              | Purpose                                       |
| --------------------- | ------------------------------------------------------ | --------------------------------------------- |
| Preprocessing         | Adaptive thresholding, CLAHE, morphological closing    | Enhance contrast, remove noise, connect lines |
| Grid Detection        | Contour approximation, Hough transform, Harris corners | Locate grid and find precise corners          |
| Perspective Transform | Homography, bilinear interpolation                     | Straighten skewed grid                        |
| OCR                   | Template matching, NCC, connected components           | Recognize digits without ML                   |
| Solver                | Backtracking with constraint checking                  | Solve valid Sudoku puzzles                    |

---

## **5. Command-Line Usage**

```bash
# Process a Sudoku image
python -m src.sudoku_solver --image puzzle.jpg --output results/

# Skip intermediate saves
python -m src.sudoku_solver --image puzzle.jpg --no-save

# Direct OCR on binary grid (skip detection)
python -m src.sudoku_solver --image binary_grid.png --binary-grid
```

---

## **6. Robustness Features**

- **Adaptive noise removal**: Detects Gaussian vs. salt-and-pepper noise, applies appropriate filters.
- **Multiple detection fallbacks**: Contour + line-based hybrid with quality scoring.
- **Grid completion**: Fills missing lines post-straightening.
- **Polarity correction**: Automatically flips white-on-black images.
- **Duplicate removal**: Resolves OCR conflicts via confidence scores.

---

## **7. Limitations & Assumptions**

1. **Grid must be visible**: The Sudoku grid should be the largest quadrilateral in the image.
2. **Moderate skew**: Extreme perspective (>45°) may fail.
3. **Digit clarity**: Handwritten or heavily degraded digits may reduce OCR accuracy.
4. **Lighting**: Works best with uniform illumination, but adapts via local normalization.

---

## **8. Dependencies**

- OpenCV ≥ 4.8.0
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0 (for optional visualization)

Install via:

```bash
pip install -r requirements.txt
```

---

## **9. Conclusion**

This project demonstrates a full computer vision pipeline for solving Sudoku puzzles from images. It combines traditional CV techniques (filtering, morphology, Hough transform, homography) with simple pattern matching for OCR, avoiding machine learning as per project requirements. The system is robust to noise, skew, and lighting variations, producing accurate solutions for clear, well-framed Sudoku images.
