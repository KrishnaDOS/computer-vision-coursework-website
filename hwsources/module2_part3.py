"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 2 Assignment Part 3 - Object Detection (Template Matching) w/ Selective Region Blurring

The purpose of this script is to find 10 objects (template matching) 
in a scene image, then blur the object regions. First, the script loads 
the scene images and folder with 10 template images. Then it iterates 
through the templates and finds the matches with OpenCV's cross correlation.
Through the module2_part1.py script, inverse filtering was used for 
eliminating overlapping boxes. The picture is displayed with the boxes 
surrounding each of the detected objects. Then with help from module2_part2.py 
the Gaussian blur is applied to each detected objetct's region.

Usage:
 - Run the script - python module2_part3.py
 - When prompted, enter the scene image's file path and template image folder's path
 - Results are displayed to user

The dependencies for the script are opencv-python, numpy, and the other module2 scripts
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import hwsources.module2_part1 as m2_1  
import hwsources.module2_part2 as m2_2 

# Locate the template files within the provided folder (only 10 template files)
def find_template_files(folder: Path, limit: int = 10) -> List[Path]:
    if not folder.exists() or not folder.is_dir(): raise FileNotFoundError(f"Folder was not found :- {folder}")
    template_files = sorted([folder / f for f in os.listdir(folder) if f.startswith("template_")])
    return template_files[:limit]

# Functionality for matching the 
def match_template_once(scene_gray: np.ndarray, tpl_gray: np.ndarray, threshold: float = 0.8):
    template_height, template_width = tpl_gray.shape[:2]
    scene_height, scene_width = scene_gray.shape[:2]
    # If the template is larger than the scene, then we skip the template
    if template_height > scene_height or template_width > scene_width:
        return [], []
    # Template matching (cross correlation)
    res = cv2.matchTemplate(scene_gray, tpl_gray, cv2.TM_CCORR_NORMED)
    max_candidates = 200
    y_idxs, x_idxs, selected_scores = m2_1.select_top_scores(res, threshold, max_candidates)
    boxes, scores = [], []
    for y, x, score in zip(y_idxs, x_idxs, selected_scores):
        boxes.append((int(x), int(y), int(x + template_width), int(y + template_height)))
        scores.append(float(score))
    return boxes, scores

# Draws outlining boxes and labels for each detected object
def draw_detections(image: np.ndarray, detections: List[Tuple[Tuple[int, int, int, int], float, str]]) -> np.ndarray:
    out = image.copy()
    for (box, score, label) in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{label}:{score:.2f}"
        cv2.putText(out, text, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return out

# Blurring each region that has been recognized as an object
def blur_regions(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], blur_multiplier: float = 2.0) -> np.ndarray:
    out = image.copy()
    # Applying blur to each region
    for (x1, y1, x2, y2) in boxes:
        x1_adjusted, y1_adjusted = max(0, x1), max(0, y1)
        x2_adjusted, y2_adjusted = min(out.shape[1], x2), min(out.shape[0], y2)
        if x2_adjusted <= x1_adjusted or y2_adjusted <= y1_adjusted: continue
        region = out[y1_adjusted:y2_adjusted, x1_adjusted:x2_adjusted] # Region is defined and stored
        # Calculating the kernel size and sigma value using region dimensions
        kernel_size, sigma = m2_2._resolve_kernel_params(region.shape, None, None)
        sigma = float(sigma) * float(blur_multiplier)
        kernel_size = max(3, int(round(kernel_size * blur_multiplier)))
        if kernel_size % 2 == 0: kernel_size += 1 # Kernel size must be odd for OpenCV
        # Application of gaussian blur for the region
        blurred_region = m2_2.apply_gaussian_blur(region, kernel_size, sigma)
        out[y1_adjusted:y2_adjusted, x1_adjusted:x2_adjusted] = blurred_region
    return out

# Main function
def main() -> None:
    if len(sys.argv) == 3:
        scene_arg = sys.argv[1]
        template_folder_arg = sys.argv[2]
    elif len(sys.argv) == 1:
        scene_arg = input("Please enter the file path for scene image :- ").strip()
        template_folder_arg = input("Please enter the file path for templates folder :- ").strip()
    else:
        print("Usage :- python module2_part3.py <scene_image_path> <templates_folder_path>")
        print("Or, run the script without any command line arguments to be prompted for file paths")
        return

    # Conversion of file path strings into Path objects
    scene_path = Path(scene_arg)
    template_folder = Path(template_folder_arg)
    if not scene_path.exists() or not scene_path.is_file(): # Validation of scene image file path
        print(f"The scene image was not found :- {scene_path}")
        return

    try: # Getting the template files from the provided folder (limiting to 10)
        template_files = find_template_files(template_folder, limit=10)
    except Exception as e:
        print(f"There was an issue finding the template files :- {e}")
        return
    if not template_files: # If no template files were found then let user know
        print(f"There were no template files found in {template_folder} (expecting files starting with 'template_').")
        return

    # Load the scene image and convert to grayscale
    scene = cv2.imread(str(scene_path))
    if scene is None:
        print(f"Failed to load scene image: {scene_path}")
        return
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    # Initialize list to store all detected objects from all templates
    all_detections: List[Tuple[Tuple[int, int, int, int], float, str]] = []

    # Set matching threshold and blur strength parameters
    threshold = 0.75
    blur_multiplier = 2.5

    # Loop through each template and find matches in the scene
    for tpl_path in template_files:
        # Load template image and convert to grayscale
        tpl = cv2.imread(str(tpl_path))
        if tpl is None:
            print(f"Skipping unreadable template: {tpl_path}")
            continue
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        # Find all locations where the template matches above the threshold
        boxes, scores = match_template_once(scene_gray, tpl_gray, threshold=threshold)

        # Skip to next template if no matches found
        if not boxes:
            continue

        # Remove overlapping detections to get the best matches
        keep_idxs = m2_1.non_max_suppression(boxes, scores, iou_threshold=0.35)
        kept_boxes = [boxes[i] for i in keep_idxs]
        kept_scores = [scores[i] for i in keep_idxs]

        # Add all kept detections to the list with their template name as label
        label = tpl_path.stem
        for b, s in zip(kept_boxes, kept_scores):
            all_detections.append((b, s, label))

    # Stop if no detections were found across all templates
    if not all_detections:
        print(f"No objects detected (threshold={threshold}).")
        return

    # Draw green boxes and labels for all detected objects
    drawn = draw_detections(scene, all_detections)

    # Extract the bounding boxes from all detections
    boxes_to_blur = [det[0] for det in all_detections]

    # Save the image with detection boxes but no blur applied
    base_no_blur, ext_no_blur = os.path.splitext(str(scene_path))
    out_no_blur = f"{base_no_blur}_matched_all{ext_no_blur}"
    cv2.imwrite(out_no_blur, drawn)
    print(f"Detections-only image saved to: {out_no_blur}")

    # Apply Gaussian blur to the detected regions
    final = blur_regions(drawn, boxes_to_blur, blur_multiplier=blur_multiplier)

    # Save the final image with detection boxes and blurred regions
    base, ext = os.path.splitext(str(scene_path))
    out_path = f"{base}_matched_all_blurred{ext}"
    cv2.imwrite(out_path, final)
    print(f"Result saved to: {out_path}")


if __name__ == "__main__":
    main()
