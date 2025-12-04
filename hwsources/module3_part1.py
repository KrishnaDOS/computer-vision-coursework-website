"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 3 Assignment Part 1 - Tasks 1-3 (Gradients, Edge/Corner Detection, Boundary Detection

The purpose of this script is to create gradient of the images 
(Laplacian of Gaussian), detect edges and corners in the images, 
and determine the exact boundaries of objects

Usage:
    1. Run the script - python module3_part1.py
    2. Select the task number once prompted
    3. Provide the folder path with the 10 images
    4. Once the images are rendered in a separate pop up window, press :-
        a. Spacebar to move to the next image
        b. s to Save
        c. q to Quit

Main dependency is numpy
"""

import cv2
import numpy as np
import glob
import os
import math

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _scaled_label_metrics(image_shape, base_scale=1.0, min_scale=0.45, max_scale=3.5, boost=1.0):
    """Return (scale, thickness, margin, line_stride) tuned to the image size."""
    height, width = image_shape[:2]
    max_dim = float(max(height, width)) or 1.0
    normalized = max_dim / 1100.0  # 1100px acts as our baseline resolution
    scale = base_scale * normalized * boost
    scale = max(min_scale, min(max_scale, scale))
    thickness = max(1, int(round(scale * 2.2)))
    margin = max(24, int(round(scale * 18)))
    line_stride = max(32, int(round(scale * 24)))
    return scale, thickness, margin, line_stride


def _draw_label(canvas, text, origin, color, scale, thickness):
    """Render stroked text so labels stay bold on any background."""
    outline_thickness = thickness + 2
    cv2.putText(canvas, text, origin, LABEL_FONT, scale, (0, 0, 0), outline_thickness, lineType=cv2.LINE_AA)
    cv2.putText(canvas, text, origin, LABEL_FONT, scale, color, thickness, lineType=cv2.LINE_AA)

# Task 1 functionality
def process_gradients_log(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale conversion
    # Using Sobel (kernel size 3) from OpenCV
    gradient_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3) # Gradient calculation (x direction)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3) # Gradient calculation (y direction)
    magnitude = cv2.magnitude(gradient_x, gradient_y) # Edge strength
    angle = cv2.phase(gradient_x, gradient_y, angleInDegrees=True) # Edge direction

    # Normalization for visualization
    magnitude_visualization = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    angle_visualization = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Laplacian of Gaussian filtered versions
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    laplacian_of_gaussian = cv2.Laplacian(blurred, cv2.CV_64F)
    # Normalization of Laplacian of Gaussian
    laplacian_of_gaussian_visualization = cv2.normalize(np.abs(laplacian_of_gaussian), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Creating canvas to display all the results together
    height, width = grayscale.shape
    canvas = np.zeros((height*2, width*2, 3), dtype=np.uint8)
    
    # BGR conversion from grayscale
    magnitude_bgr = cv2.cvtColor(magnitude_visualization, cv2.COLOR_GRAY2BGR)
    angle_bgr = cv2.cvtColor(angle_visualization, cv2.COLOR_GRAY2BGR)
    laplacian_of_gaussian_bgr = cv2.cvtColor(laplacian_of_gaussian_visualization, cv2.COLOR_GRAY2BGR)

    # Arranging images into a 2 x 2 grid layout
    canvas[0:height, 0:width] = image
    canvas[0:height, width:2*width] = magnitude_bgr
    canvas[height:2*height, 0:width] = angle_bgr
    canvas[height:2*height, width:2*width] = laplacian_of_gaussian_bgr
    label_scale, label_thickness, margin, _ = _scaled_label_metrics(image.shape, base_scale=1.15, min_scale=0.7, max_scale=4.2, boost=1.35)
    x_margin = max(12, int(round(label_scale * 12)))
    top_y = margin
    bottom_y = height + margin
    # Text labels sized relative to the source resolution
    _draw_label(canvas, "Original Image", (x_margin, top_y), (0, 0, 255), label_scale, label_thickness)
    _draw_label(canvas, "Gradient Magnitude", (width + x_margin, top_y), (0, 0, 255), label_scale, label_thickness)
    _draw_label(canvas, "Gradient Angle", (x_margin, bottom_y), (0, 0, 255), label_scale, label_thickness)
    _draw_label(canvas, "Laplacian of Gaussian", (width + x_margin, bottom_y), (0, 0, 255), label_scale, label_thickness)
    return canvas

# Task 2 functionality
def detect_features(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale conversion
    output = image.copy()
    # Using Sobel (kernel size 3) from OpenCV
    gradient_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3) # Gradient calculation (x direction)
    gradient_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3) # Gradient calculation (y direction)
    magnitude = cv2.magnitude(gradient_x, gradient_y) # Edge strength
    edge_threshold = np.percentile(magnitude, 85)
    edges = magnitude > edge_threshold
    output[edges] = [0, 255, 0]
    # Using absolute values for corner detection
    gradient_x = np.abs(gradient_x) 
    gradient_y = np.abs(gradient_y)
    
    # Normalization process
    gradient_x_normalized = cv2.normalize(gradient_x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gradient_y_normalized = cv2.normalize(gradient_y, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # We are using top 20% instead of top 15% so more corners are detected
    gradient_x_threshold = np.percentile(gradient_x_normalized, 80)
    gradient_y_threshold = np.percentile(gradient_y_normalized, 80)
    # Gradients x and gradients y have to be above their respective thresholds which will distinguish corners from edges
    corners = (gradient_x_normalized > gradient_x_threshold) & (gradient_y_normalized > gradient_y_threshold)
    corners = corners & ~edges # Making sure edges aren't marked as corners
    output[corners] = [0, 0, 255]
    annot_scale, annot_thickness, margin, line_stride = _scaled_label_metrics(image.shape, base_scale=0.85, min_scale=0.55, max_scale=3.0, boost=1.2)
    x_margin = max(12, int(round(annot_scale * 12)))
    first_line = margin
    second_line = margin + line_stride
    _draw_label(output, "Edges (green) :- higher gradient magnitude", (x_margin, first_line), (0, 255, 0), annot_scale, annot_thickness)
    _draw_label(output, "Corners (red) :- higher gradient x and gradient y", (x_margin, second_line), (0, 0, 255), annot_scale, annot_thickness)
    return output

# Task 3 functionality
def find_exact_boundary(image):
    denoised = cv2.bilateralFilter(image, 7, 50, 50)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(l_channel)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(cv2.countNonZero(otsu_mask)) / max(1, otsu_mask.size)
    if white_ratio > 0.5:
        otsu_mask = cv2.bitwise_not(otsu_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opened, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers.astype(np.int32) + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(denoised.copy(), markers)

    region_mask = np.zeros_like(otsu_mask)
    best_label = None
    best_area = 0
    image_area = image.shape[0] * image.shape[1]
    for label in np.unique(markers):
        if label <= 1:
            continue
        component = np.uint8(markers == label)
        area = cv2.countNonZero(component)
        if area > best_area and area > 0.002 * image_area:
            best_area = area
            best_label = label
    if best_label is not None:
        region_mask[markers == best_label] = 255
    else:
        fallback_contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if fallback_contours:
            fallback = max(fallback_contours, key=cv2.contourArea)
            cv2.drawContours(region_mask, [fallback], -1, 255, thickness=cv2.FILLED)

    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    gc_mask = np.full(region_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    fg_seed = region_mask == 255
    bg_seed = sure_bg == 0
    border = 10
    bg_seed[:border, :] = True
    bg_seed[-border:, :] = True
    bg_seed[:, :border] = True
    bg_seed[:, -border:] = True
    gc_mask[bg_seed] = cv2.GC_BGD
    gc_mask[fg_seed] = cv2.GC_FGD
    gc_mask[sure_fg == 255] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(denoised, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        pass

    grabcut_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    grabcut_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    grabcut_mask = cv2.morphologyEx(grabcut_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(grabcut_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()
    contour_color = (0, 140, 255)
    primary = None
    if contours:
        primary = max(contours, key=cv2.contourArea)
    elif cv2.countNonZero(region_mask) > 0:
        fallback_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if fallback_contours:
            primary = max(fallback_contours, key=cv2.contourArea)
            grabcut_mask = region_mask.copy()

    if primary is not None:
        cv2.drawContours(overlay, [primary], -1, contour_color, thickness=3)
        tint = np.zeros_like(overlay)
        tint[:, :, 1] = grabcut_mask
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.4, 0)

    edge_map = cv2.Canny(grabcut_mask, 50, 150)
    watershed_boundaries = np.uint8(markers == -1) * 255
    edge_map = cv2.bitwise_or(edge_map, watershed_boundaries)

    mask_bgr = cv2.cvtColor(grabcut_mask, cv2.COLOR_GRAY2BGR)
    edges_bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

    height, width = image.shape[:2]
    canvas = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    canvas[0:height, 0:width] = image
    canvas[0:height, width:2 * width] = overlay
    canvas[height:2 * height, 0:width] = mask_bgr
    canvas[height:2 * height, width:2 * width] = edges_bgr

    label_scale, label_thickness, margin, line_stride = _scaled_label_metrics(
        image.shape, base_scale=0.95, min_scale=0.6, max_scale=3.6, boost=1.3
    )
    x_margin = max(12, int(round(label_scale * 12)))
    top_y = margin
    bottom_y = height + margin

    _draw_label(canvas, "Original Image", (x_margin, top_y), (0, 0, 255), label_scale, label_thickness)
    _draw_label(canvas, "Refined Boundary Overlay", (width + x_margin, top_y), contour_color, label_scale, label_thickness)
    _draw_label(canvas, "GrabCut Object Mask", (x_margin, bottom_y), (255, 255, 255), label_scale, label_thickness)
    _draw_label(canvas, "Edge & Watershed Lines", (width + x_margin, bottom_y), (0, 255, 255), label_scale, label_thickness)

    if primary is None:
        warning_y = bottom_y + line_stride
        _draw_label(canvas, "Warning: no stable boundary detected", (x_margin, warning_y), (0, 0, 255), label_scale * 0.85, label_thickness)

    return canvas

# Main function
def main():
    images = []    
    print("Module 3 (tasks 1- 3)")
    print("Please select one of the following options")
    print("1. Gradients & Laplacian of Gaussian")
    print("2. Keypoints detection (edges & corners)")
    print("3. Exact boundary detection")
    
    # Make sure the user selection is valid
    while True:
        try:
            task_input = input("Enter Task (numbers 1, 2, or 3 are valid selections) :- ").strip()
            if not task_input: continue # Empty input case
            task_number = int(task_input)
            if 1 <= task_number <= 3: break# Valid task number was entered by user
            else: print("Please enter numbers 1, 2, or 3")
        except ValueError: print("Please enter numbers 1, 2, or 3") 
    
    while True:
        src = input("Please enter the image / folder path :- ").strip()
        if os.path.exists(src):
            if os.path.isfile(src): # Single image case
                images = [src]
            else: # Folder / multiple images case (main one)
                files = sorted(glob.glob(os.path.join(src, "*.*")))
                images = [x for x in files if x.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif'))]
                if not images: # If folder did not contain any valid image files
                    print("Folder does not contain any valid image files")
                    continue
            break
        else: 
            print("Path to image / folder was invalid. Please make sure to enter a correct path")

    print(f"\nRunning the task number {task_number}")
    print("Press spacebar to move to next picture, s to save, and q to quit program")

    for image_path in images:
        frame = cv2.imread(image_path)
        if frame is None: continue # Skip any corrupt frames
        # Routing the image to the proper function based on the user input
        if task_number == 1: result = process_gradients_log(frame)
        elif task_number == 2: result = detect_features(frame)
        elif task_number == 3: result = find_exact_boundary(frame)
        cv2.imshow(f"Task number {task_number}", result) # Display window
        
        stop = False
        while True:
            # Wait for the key selection
            k = cv2.waitKey(0) & 0xFF            
            if k == 32: break  # If spacebar, move to next image
            if k == ord('q'): stop = True; break  # q means quit the program
            if k == ord('s'): # s means save the result
                out_name = f"out_task{task_number}_{os.path.basename(image_path)}"
                cv2.imwrite(out_name, result)
                print(f"File has been saved to {out_name}")
        
        if stop: break

    # Clean up all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()