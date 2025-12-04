"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 4 Assignment Part 1 - Image Stitching

The purpose of this script is to perform image stitching 
on 8 portrait orientation images, assuming those 8 images 
are overlapping. OpenCV's SIFT feature detection has been 
used for identification of key points in the overlapping 
images. Image rotation, scaling and translation were used 
for preventing distortion in the final result. Mask erosion 
was also added additionally to remove any thin black lines 
that were present in the border of the individual images.

Usage:
    1. Images should be named in ascending order (left to right, 1 to 8) and placed together in a folder
    2. Run the script - python module4_part1.py [optional/path/to/images]
    3. You can pass an absolute path or relative path to the image folder
    4. Script will process the images and stitch them together to create a panorama image
    5. The final result is stored as stitched_result_final.jpg and displayed to the user in a pop up window
"""

import cv2
import numpy as np
import glob
import os
import sys
import re

# Global constants
WORK_WIDTH = 800 # Images will be downscaled to this work width for making the feature detection process faster
SEAM_EROSION_ITERATIONS = 1 # Will be used for removing the thin black border lines at the edges of each picture

sys.setrecursionlimit(2000) # Increasing allowed depth for recusion (directory traversal, etc)

# Image resizing helper method
def resize_image(image, target_width=None):
    if target_width is None: return image # If user has not defined a target width, then return immediately
    (original_height, original_width) = image.shape[:2]
    ratio = target_width / float(original_width)
    dimensions = (target_width, int(original_height * ratio))
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA) # INTER_AREA is best especially when shrinking the image

# Sorts numbers strings in proper order instead of first digit order (1, 2, 10 vs. 1, 10, 2)
def correct_number_sorting(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# Extracts SIFT keypoints and descriptors from an image
def extract_sift_keypoints_descriptors(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale conversion
    sift = cv2.SIFT_create(contrastThreshold=0.03) # Will capture features in areas of low contrast (e.g. grass/sky)
    keypoints, features = sift.detectAndCompute(grayscale, None)
    return keypoints, features

# Matches keypoints and features and calculates Affine transformation matrix
def match_keypoints_affine(keypointsA, keypointsB, featuresA, featuresB, ratio=0.75):
    matches = []
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(featuresA, featuresB, k=2) # Retrieving the top k matches
    # Keep the "best match" only if there's enough difference between "best" and "second best" matches
    for best_match, second_best_match in raw_matches:
        if best_match.distance < ratio * second_best_match.distance:
            matches.append(best_match)

    if len(matches) > 4: # To compute the Affine transformation we need at least 4 matches
        pointsA = np.float32([keypointsA[match.queryIdx].pt for match in matches]) # Getting the points location from matches
        pointsB = np.float32([keypointsB[match.trainIdx].pt for match in matches])
        matrix, _ = cv2.estimateAffinePartial2D(pointsA, pointsB) # Estimation of affine transformation (rotation, translation, scaling)
        if matrix is None: return None
        homography = np.vstack([matrix, [0, 0, 1]]) # Conversion of 2x3 affine matrix to 3x3 homography format
        status = np.ones(len(matches), dtype=np.uint8)
        return (matches, homography, status)
    return None

# Removing the black borders from the stitched image
def remove_black_borders(panorama):
    if panorama is None or panorama.size == 0: return None # If panorama is null or invalid
    grayscale = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY) # Grayscale conversion
    _, threshold = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY) # Creating the binary mask (non-black pixel)
    kernel = np.ones((5,5), np.uint8) #Dilating the mask a little to make sure valid dark edges aren't removed
    threshold = cv2.dilate(threshold, kernel, iterations=2)
    coordinates = cv2.findNonZero(threshold) # Get all the coordinate pairs of valid pixels to find the crop area
    if coordinates is None: return panorama
    x, y, width, height = cv2.boundingRect(coordinates)
    return panorama[y:y+height, x:x+width]

# Main function
def main():
    if len(sys.argv) > 1: # Handling CLI arguments or regular user input
        provided_path = sys.argv[1]
    else:
        provided_path = input("Please enter the image folder path (8 portrait images) :- ").strip()

    # Handling relative paths, current working directory, or script directory
    provided_path = os.path.expanduser(provided_path)
    if provided_path == "": provided_path = os.getcwd()
    image_folder = os.path.abspath(provided_path)

    # Checking if the path exists
    if not os.path.isdir(image_folder):
        script_directory = os.path.dirname(os.path.abspath(__file__))
        alt_image_directory = os.path.abspath(os.path.join(script_directory, provided_path))
        if os.path.isdir(alt_image_directory):
            image_folder = alt_image_directory
        else:
            print(f"The folder was not found at this path :- {image_folder}")
            return

    # Loading all the images
    print(f"Loading the images from the folder {image_folder}")
    image_paths = glob.glob(os.path.join(image_folder, "*"))
    image_paths.sort(key=correct_number_sorting) # Sorting all the files    
    original_images = [cv2.imread(p) for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    original_images = [image for image in original_images if image is not None]
    
    # Should be at least 4 images
    if len(original_images) < 4:
        print("Please provide at least 4 images (landscape) or 8 (portrait)")
        return
    print(f"Processing {len(original_images)} images...")

    # Resizing the images to reduce computational costs
    images = [resize_image(image, target_width=WORK_WIDTH) for image in original_images]
    homographies = [np.identity(3)] # Initial homographies - identity matrix for first image
    for i in range(len(images) - 1): # Calculation of relative homographies between adjacent pairs
        print(f"Aligning the image {i+2} with {i+1}")
        keypoints_previous, feats_previous = extract_sift_keypoints_descriptors(images[i])
        keypoints_current, feats_current = extract_sift_keypoints_descriptors(images[i+1])
        result = match_keypoints_affine(keypoints_current, keypoints_previous, feats_current, feats_previous)
        if result is None:
            print(f"Unable to align the image {i+1} with {i+2}")
            return
        # Chaining the homographies together relative to first image
        (_, homography_current_to_previous, _) = result
        homographies.append(homographies[i].dot(homography_current_to_previous))

    # Recalculating the homographies relative to the center image - minimizing distortions
    middle_index = len(images) // 2
    homography_middle_to_0 = homographies[middle_index]
    homography_0_to_middle = np.linalg.inv(homography_middle_to_0)
    new_homographies = []
    all_corners = []
    
    # Transforming all images' corners to determine total size of final panorama
    for i, homography in enumerate(homographies):
        new_homography = homography_0_to_middle.dot(homography) # Retarget to center image
        new_homographies.append(new_homography)
        height, width = images[i].shape[:2]
        corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, new_homography))

    # Calculation of canvas size
    print("Calculating canvas size")
    all_corners = np.concatenate(all_corners, axis=0)
    x_minimum, y_minimum = np.int32(all_corners.min(axis=0).ravel())
    x_maximum, y_maximum = np.int32(all_corners.max(axis=0).ravel())
    output_width, output_height = x_maximum - x_minimum, y_maximum - y_minimum
    print(f"Canvas size will be :- {output_width} x {output_height}")

    print("Stitching images together")
    # Shifting any of the negative coordinate values into positive valuses
    homography_translation = np.array([[1, 0, -x_minimum], [0, 1, -y_minimum], [0, 0, 1]])
    panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    erosion_kernel = np.ones((3, 3), np.uint8) # 3x3 matrix used to shave the outer edge
    # Order is to draw outer images first and move towards the center
    # Middle image (also the least distorted one) is placed last on top
    draw_order = sorted(range(len(images)), key=lambda x: abs(x - middle_index), reverse=True)

    # Iterating through the images in the order defined earlier
    for i in draw_order:
        image = images[i]
        final_homography = homography_translation.dot(new_homographies[i]) # Combination of image alignment matrix and canvas centering matrix
        warped = cv2.warpPerspective(image, final_homography, (output_width, output_height), flags=cv2.INTER_LINEAR) # Transforming image from local coordinates into final panorama's coordinates
        grayscale = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) # Grayscale conversion
        _, mask = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY) # # Generating binary mask of the warped & grayscaled image
        mask = cv2.erode(mask, erosion_kernel, iterations=SEAM_EROSION_ITERATIONS) # Getting rid of 1 px from the edges (removing dark border lines)
        panorama[mask > 0] = warped[mask > 0] # Overlaying the proper pixels from the image into the panorama

    # Removing the black borders from the edges
    print("Trimming panorama image")
    panorama = remove_black_borders(panorama)
    
    # Saving the final panorama
    output_path = os.path.join(image_folder, "stitched_result_final.jpg")
    cv2.imwrite(output_path, panorama)
    print(f"Final panorama image has been saved to :- {output_path}")

if __name__ == "__main__":
    main()