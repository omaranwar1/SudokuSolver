#!/usr/bin/env python3
"""
Process all Sudoku images and generate OCR detection visualizations.
"""

import sys
import os
import glob

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.sudoku_solver import SudokuSolver


def main():
    """Process all .jpg images in the current directory."""
    # Find all jpg files
    image_files = sorted(glob.glob("*.jpg"))
    
    if not image_files:
        print("No .jpg files found in current directory!")
        return
    
    print(f"Found {len(image_files)} images to process")
    print("=" * 60)
    
    solver = SudokuSolver(output_size=450, save_intermediate=True)
    output_dir = "output"
    
    results = {
        'solved': [],
        'unsolved': [],
        'error': []
    }
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_path}...")
        
        try:
            result = solver.process_image(image_path, output_dir)
            if result:
                # Check if solution was found
                if 'solved_overlay' in solver.intermediate_images:
                    results['solved'].append(image_path)
                else:
                    results['unsolved'].append(image_path)
            else:
                results['error'].append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results['error'].append(image_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Solved:    {len(results['solved'])}/{len(image_files)}")
    print(f"❌ Unsolved:  {len(results['unsolved'])}/{len(image_files)}")
    print(f"⚠️  Errors:    {len(results['error'])}/{len(image_files)}")
    
    if results['solved']:
        print(f"\nSolved images: {', '.join(results['solved'])}")
    
    print(f"\nOCR detection visualizations saved to: {output_dir}/")
    print("Look for files named: XX_ocr_detection.jpg")


if __name__ == '__main__':
    main()
