# ✨ Sudoku Solver — Computer Vision Project

**Course:** CSCE 4603 — Fundamentals of Computer Vision
**Project Weight:** 30%
**Team Member:** **Omar Anwar** **Farida Bey** **Amal Fouda** **Ahmed El Barbary**

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

### Advanced Options

```
python process_image.py --image 01.jpg --output results/
python process_image.py --image 01.jpg --size 600
python process_image.py --image 01.jpg --no-save
```

### Output (Generated Automatically)

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
