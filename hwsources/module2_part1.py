"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 2 Assignment Part 1 - Object Detection through Correlation-Based Template Matching

The purpose of this script is to detect a given 
object (template) within an image (scene). For this 
part of the assignment, OpenCV's template matching 
functionality has been used. There were issues with 
identification due to the lighting differences between 
template and scene. As such I had to use OpenCV's normalized 
cross correlation mapping for getting accurate correlation scores.

Usage:
	1. Run the script - python module2_part1.py
    2. Enter the file path of the scene image
    3. Enter the file path of template image
    4. Script will save an output image with a 
       box drawn around the detected object if found

Notes:
	For better results make sure template image is close 
    in terms of scale and well as rotation/orientation 
    to the object in the scene.

Dependencies include OpenCV (cv2) and NumPy
"""

from __future__ import annotations
import os
from typing import List, Tuple
import cv2
import numpy as np
import sys

# Suppress overlapping boxes
def non_max_suppression(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_threshold: float = 0.4):
    if not boxes: return []

    boxes_arr = np.array(boxes, dtype=float)
    scores_arr = np.array(scores, dtype=float)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_arr.argsort()[::-1]
    keep = []

    # Retain the highest correlation score
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width * height
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]

    return keep


def select_top_scores(score_map: np.ndarray, threshold: float, max_candidates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the row indices, column indices, and scores for the strongest template matches."""
    if max_candidates <= 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    hit_mask = score_map >= threshold
    hit_count = int(hit_mask.sum())
    if hit_count == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    if hit_count <= max_candidates:
        y_idxs, x_idxs = np.where(hit_mask)
        scores = score_map[y_idxs, x_idxs]
        order = np.argsort(scores)[::-1]
        return y_idxs[order], x_idxs[order], scores[order]

    flat_scores = score_map.reshape(-1)
    limit = min(max_candidates * 4, flat_scores.size)
    idxs = np.argpartition(flat_scores, -limit)[-limit:]
    idxs = idxs[np.argsort(flat_scores[idxs])[::-1]]
    y_idxs, x_idxs = np.unravel_index(idxs, score_map.shape)
    scores = flat_scores[idxs]

    valid = scores >= threshold
    y_idxs, x_idxs, scores = y_idxs[valid], x_idxs[valid], scores[valid]
    if y_idxs.size > max_candidates:
        y_idxs, x_idxs, scores = y_idxs[:max_candidates], x_idxs[:max_candidates], scores[:max_candidates]

    return y_idxs.astype(np.int32), x_idxs.astype(np.int32), scores.astype(np.float32)

# Template matching
def match_template(scene_path: str, template_path: str, threshold: float = 0.8, save_result: bool = True, *, draw_all: bool = True) -> bool:
    if not os.path.isfile(scene_path): raise FileNotFoundError(f"Scene file not found :- {scene_path}")
    if not os.path.isfile(template_path): raise FileNotFoundError(f"Template file not found :- {template_path}")

    # Load the images if paths are correct
    scene = cv2.imread(scene_path)
    template = cv2.imread(template_path)
    if scene is None: raise ValueError(f"Unable to load scene image :- {scene_path}")
    if template is None: raise ValueError(f"Unable to load template image :- {template_path}")

    # Conversion to grayscale so we can analyze brightness levels accurately
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_height, template_width = template_gray.shape[:2]
    scene_height, scene_width = scene_gray.shape[:2]
    if template_height > scene_height or template_width > scene_width: raise ValueError("Template image is much larger than the scene object")

    # Using OpenCV cross-correlation template matching
    score_map = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCORR_NORMED)
    max_candidates = 50
    y_idxs, x_idxs, selected_scores = select_top_scores(score_map, threshold, max_candidates)

    # Only continue if there are any candidates
    if y_idxs.size == 0:
        print(f"No matches found  with the threshold set to :- {threshold}")
        return False

    candidate_boxes: List[Tuple[int, int, int, int]] = []
    candidate_scores: List[float] = []
    for y, x, score in zip(y_idxs, x_idxs, selected_scores):
        candidate_boxes.append((int(x), int(y), int(x + template_width), int(y + template_height)))
        candidate_scores.append(float(score))

    # Application of non-maximum suppression for removing overlapping detections
    keep_idxs = non_max_suppression(candidate_boxes, candidate_scores, iou_threshold=0.35)
    final_boxes = [candidate_boxes[i] for i in keep_idxs]
    final_scores = [candidate_scores[i] for i in keep_idxs]

    out = scene.copy()
    if not draw_all and final_scores:
        best_index = int(np.argmax(final_scores))
        final_boxes = [final_boxes[best_index]]
        final_scores = [final_scores[best_index]]

    best_index = 0
    if final_scores:
        best_index = int(np.argmax(final_scores))

    # Draw the boxes with their respective scores
    for (box, score) in zip(final_boxes, final_scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{score:.2f}"
        cv2.putText(out, text, (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # Emphasizing the best match for quickly pruning
    if final_boxes:
        box = final_boxes[best_index]
        bscore = final_scores[best_index]
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 200), 3)
        label = f"Best {bscore:.2f}"
        cv2.putText(out, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

    # If this flag was set, then the result is saved to a file
    if save_result:
        base, extension = os.path.splitext(scene_path)
        template_name = os.path.splitext(os.path.basename(template_path))[0]
        output_file_path = f"{base}_matched_{template_name}{extension}"
        cv2.imwrite(output_file_path, out)
        print(f"Result has been successfully saved to :- {output_file_path}")

    if draw_all:
        print(f"These many matches were found with threshold set to {threshold} :- {len(final_boxes)}")
    else:
        print(f"1 best match was found with threshold set to {threshold} :- {final_scores[0]:.2f}")
    return True

# Main function
def main() -> None:
    print("Template-matching detection (correlation-based)\n")
    print("Provide the full path (or relative to the repo root) for both :-")
    print(" - the scene image that contains the object to search for")
    print(" - the template image containing the object to match")
    print()

    # Helper method to resue the prompt 
    def prompt_for_path(prompt_text: str) -> str:
        while True:
            p = input(prompt_text).strip()
            if p and os.path.isfile(p):
                return p
            print("Path is missing or file does not exist - please give a correct file path")

    # Threshold value
    threshold = 0.6

    # Allow some optional scene and template arguments
    if len(sys.argv) >= 3:
        scene_path = sys.argv[1]
        tpl_arg = sys.argv[2]

        if not os.path.isfile(scene_path):
            print(f"Scene file was not found :- {scene_path}")
            return
        if os.path.isdir(tpl_arg):
            found_any = False
            matches_overall = False
            template_files = sorted([os.path.join(tpl_arg, f) for f in os.listdir(tpl_arg) if f.startswith("template_")])
            if not template_files:
                print(f"No template files were found in the folder {tpl_arg} (searched for files starting with 'template_')")
                return

            print(f"These many template files were found in the folder {tpl_arg} :- {len(template_files)}. Running the matches")
            matched_template_counter = int(0)
            for template_file in template_files:
                try: # Try each of the template files and see if any match is found
                    matched = match_template(scene_path, template_file, threshold, save_result=True)
                    found_any = True
                    if matched:
                        matches_overall = True
                        print(f"This template file was matched with the scene {scene_path} :- {template_file}")
                        matched_template_counter += 1
                    else:
                        print(f"No match was found with the template file {template_file}")
                except Exception as e:
                    print(f"Template file {template_file} has been skipped due to some issue :- {e}")

            if not found_any:
                print("No templates were found/processed")
            elif matches_overall:
                print(f"These many templates were matched with the scene {scene_path} :- {matched_template_counter}")
            else:
                print("No templates were matched the scene")
        else:
            if not os.path.isfile(tpl_arg):
                print(f"No template found :- {tpl_arg}")
                return

            print("Trying the matching between template object and scene")
            try:
                matched = match_template(scene_path, tpl_arg, threshold=threshold, save_result=True)
                if matched:
                    print(f"Object was detected in the scene {scene_path}")
                else:
                    print(f"Object was not detected in the scene {scene_path}")
            except Exception as e:
                print(f"There was an issue :- {e}")
    
    # If no command line arguments were provided 
    else:
        def prompt_for_path(prompt_text: str) -> str:
            while True:
                p = input(prompt_text).strip()
                if p and os.path.isfile(p):
                    return p
                print("Path is missing or file does not exist - please give a correct file path")

        scene_path = prompt_for_path("Enter the scene file path :- ")
        template_path = prompt_for_path("Enter the template file path :- ")
        print("Trying the matching between template object and scene")

        try:
            matched = match_template(scene_path, template_path, threshold=threshold, save_result=True)
            if matched:
                print(f"Object was detected in the scene {scene_path}")
            else:
                print(f"Object was not detected in the scene {scene_path}")
        except Exception as e:
            print(f"There was an issue :- {e}")

if __name__ == "__main__":
    main()