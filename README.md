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
- **Outer frame isolation** (4 marks) - In Progress
- **Outer frame corners identification** (2 marks) - Pending
- **Grid straightening into a square** (2 marks) - Pending

## Current Implementation 

### ✅ Image Preprocessing Module (4 marks)

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

### Current Functionality (Branch 1)

```bash
# Test preprocessing on an image
python -c "from src.preprocessing import preprocess_image; preprocess_image('01.jpg', display=True)"
```

This will display the preprocessing stages and save intermediate results.

## Project Structure

```
SudokuSolver/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Entry point for module execution
│   └── preprocessing.py         # Image preprocessing (Milestone 1: 4 marks) ✅
├── process_image.py             # Convenience script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Next Steps

Grid detection and corner identification (6 marks)
- Implement contour-based detection
- Implement line-based Hough Transform detection
- Adaptive method selection
- Corner extraction and validation

Perspective transformation (2 marks)
- Homography computation
- Grid straightening to 450×450

Pipeline integration
- Comprehensive output

**Current Status**: Preprocessing Complete (4/12 marks)
**Next Milestone**: Milestone 2 - December 4, 2025
