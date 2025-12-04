"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 4 Assignment Part 2 - Image Stitching with SIFT and RANSAC (from scratch)

The purpose of the script is to perform image 
stitching on a sequence of overlapping portrait 
orientation images, the difference being SIFT and 
RANSAC were implemented from scratch

Some key details for the implementation :-
 - SIFT based on difference of Gaussian
 - RANSAC I used affine transformation based approach instead of homography
 - There was bowtie distortion, so I have applied centering based on whichever 
   image is in the middle of the sequence
 - Comparison is performed between manual SIFT/RANSAC implementation, as well as 
   OpenCV's SIFT/RANSAC methods

Usage:
    1. Images should be named in ascending order (left to right, 1 to 8) and placed in a folder
    2. Run the script - python module4_part2.py
    3. Enter the file path of the image folder
    4. The script will run the images through OpenCV's SIFT/RANSAC as well as the custom SIFT/RANSAC
    5. Final results are saved
"""

import cv2
import numpy as np
import glob
import os
import re

# Global constants
WORK_WIDTH = 400 # Will be used to downscale images so computational cost will reduce 

# Custom SIFT implementation
class CustomSIFT:
    # Initialization function
    def __init__(self, sigma=1.6, intervals=3, contrast_threshold=0.04, edge_threshold=10):
        self.sigma = sigma
        self.intervals = intervals
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold

    # Get the Gaussian images - each image is progressively blurred
    def generate_gaussian_images(self, image, sigma):
        gaussian_images = [image] # Storing the image first
        k = 2 ** (1 / self.intervals) # Blurring scale factor
        # Generating self.intervals + 3 images so that there are enough extra layers on top/bottom for keypoint detection
        for i in range(self.intervals + 2):
            current_sigma = (k**i) * sigma #Calculation of sigma for current iteration
            blur = cv2.GaussianBlur(image, (0, 0), current_sigma) # Applying gaussian blur
            gaussian_images.append(blur)
        return gaussian_images

    # Getting the Difference of Gaussian for Laplacian approximation
    def generate_difference_of_gaussian(self, gaussian_images):
        difference_of_gaussian_images = []
        # Finding features by subtracting adjacent images
        for i in range(len(gaussian_images) - 1): difference_of_gaussian_images.append(gaussian_images[i+1] - gaussian_images[i])
        return difference_of_gaussian_images

    # Used to determine whether a feature appears to be an edge
    def is_edge_like(self, image, row, column):
        value = image[row, column]
        # Calculate various gradients - second derivatives Dxx, Dyy, Dxy
        Dxx = image[row, column + 1] + image[row, column - 1] - 2 * value
        Dyy = image[row + 1, column] + image[row - 1, column] - 2 * value
        Dxy = (image[row + 1, column + 1] + image[row - 1, column - 1] - image[row - 1, column + 1] - image[row + 1, column - 1]) / 4.0
        trace = Dxx + Dyy
        determinant = Dxx * Dyy - Dxy**2
        if determinant <= 0: return True # If determinant is <= 0 then feature is an edge
        score = (trace ** 2) / determinant
        threshold = ((self.edge_threshold + 1) ** 2) / self.edge_threshold
        return score >= threshold

    # Finds keypoints within the Difference of Gaussian images
    def find_keypoints(self, difference_of_gaussian_images):
        keypoints = []
        height, width = difference_of_gaussian_images[0].shape
        for i in range(1, len(difference_of_gaussian_images) - 1): # Iterating through each blurring layer excluding first/last layer
            previous, current, next = difference_of_gaussian_images[i - 1], difference_of_gaussian_images[i], difference_of_gaussian_images[i + 1]
            # Skipping the 15 px border so that the 16 x 16 descriptor window doesn't fall off thhe edge
            for row in range(15, height - 15):
                for column in range(15, width - 15):
                    value = current[row, column]
                    if abs(value) < self.contrast_threshold: continue # Removing weak features (not enough contrast)
                    # Comparing the pixel against 9 neighbors in previous layer, 8 in current layer, and 9 in next layer
                    block = np.array([previous[row - 1 : row + 2, column - 1 : column + 2], current[row - 1 : row + 2, column - 1 : column + 2], next[row - 1 : row + 2, column - 1 : column + 2]])
                    if value == block.max() or value == block.min(): # Checking if the pixel is maximum or minimum value among the 26 neighbors
                        # Remove points on edges that are not corners
                        if not self.is_edge_like(current, row, column): keypoints.append(cv2.KeyPoint(float(column), float(row), size=10))
        return keypoints

    # Calculating gradients for each coordinate by using the surrounding 4 neighbors
    def get_interpolated_gradients(self, row, column, dx_image, dy_image):
        # Identifying surrounding 4 neighbors
        row0, column0 = int(np.floor(row)), int(np.floor(column))
        row1, column1 = row0 + 1, column0 + 1
        # If the 2x2 grid falls outside of image limits
        if row0 < 0 or column0 < 0 or row1 >= dx_image.shape[0] or column1 >= dx_image.shape[1]: return 0.0, 0.0
        row_weight, column_weight = row - row0, column - column0 # Calculation of weights
        # Bilinear interpolation - sum of (neighbor's value * weight) for all 4 corners
        dx = ((1 - row_weight) * (1 - column_weight) * dx_image[row0, column0]) + ((1 - row_weight) * column_weight * dx_image[row0, column1]) + (row_weight * (1 - column_weight) * dx_image[row1, column0]) + (row_weight * column_weight * dx_image[row1, column1])
        dy = ((1 - row_weight) * (1 - column_weight) * dy_image[row0, column0]) + ((1 - row_weight) * column_weight * dy_image[row0, column1]) + (row_weight * (1 - column_weight) * dy_image[row1, column0]) + (row_weight * column_weight * dy_image[row1, column1])
        return dx, dy

    # Getting the 128-dimensional descriptors for the keypoints
    def compute_descriptor(self, image, keypoints):
        descriptors = []
        final_keypoints = []
        image = image.astype(np.float32)
        dx_image = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1) # Computing gradients
        dy_image = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
        image_magnitude, image_angle = cv2.cartToPolar(dx_image, dy_image, angleInDegrees=True)
        
        # Calculation of dominant gradient direction for each keypoint (for rotation invariance)
        for keypoint in keypoints:
            column, row = int(keypoint.pt[0]), int(keypoint.pt[1])
            histogram = np.zeros(36) # 36 bins for the histogram
            for i in range(-4, 5):
                for j in range(-4, 5):
                    if 0 <= row + i < image.shape[0] and 0 <= column + j < image.shape[1]:
                        bin_index = int(image_angle[row + i, column + j] / 10) % 36
                        histogram[bin_index] += image_magnitude[row + i, column + j]
            keypoint.angle = float(np.argmax(histogram) * 10)
            final_keypoints.append(keypoint)

        # Creating 2D (16x16) Gaussian window - multiplying 1D kernel with its transpose
        gaussian_weight = cv2.getGaussianKernel(16, 8) @ cv2.getGaussianKernel(16, 8).T
        valid_keypoints = []
        
        # Iterating through each keypoint to create feature vector
        for keypoint in final_keypoints:
            keypoint_column, keypoint_row, keypoint_angle = keypoint.pt[0], keypoint.pt[1], keypoint.angle
            angle_radian = np.deg2rad(-keypoint_angle)
            cos_angle, sin_angle = np.cos(angle_radian), np.sin(angle_radian)
            descriptor_vector = np.zeros((4, 4, 8))
            valid_window = True
            
            # Looping through each pixel in the descriptor window
            for i in range(16):
                for j in range(16):
                    # Coordinate rotation - centering the grid (-7.5 to 7.5 ) and rotation
                    row_grid, column_grid = i - 7.5, j - 7.5
                    column_rotation = column_grid * cos_angle - row_grid * sin_angle
                    row_rotation = column_grid * sin_angle + row_grid * cos_angle
                    #Finding absolute subpixel coordinate in image
                    sample_row, sample_column = keypoint_row + row_rotation, keypoint_column + column_rotation
                    # Gradient calculation
                    dx, dy = self.get_interpolated_gradients(sample_row, sample_column, dx_image, dy_image)
                    if dx == 0 and dy == 0 and (sample_row < 2 or sample_row > image.shape[0] - 2):
                        valid_window = False; break
                        
                    # Getting relative gradient features
                    value_magnitude = np.sqrt(dx**2 + dy**2)
                    value_angle = np.degrees(np.arctan2(dy, dx)) % 360
                    relative_angle = (value_angle - keypoint_angle) % 360 # SUbtracting keypoint angle for rotation invariance
                    weight = value_magnitude * gaussian_weight[i, j] # Gaussian weights

                    # Mapping each pixel to 4x4 grid and 8 orientation bins
                    row_bin, column_bin, orientation_bin = (i + 0.5)/4.0 - 0.5, (j + 0.5)/4.0 - 0.5, relative_angle/45.0
                    row0, column0, orientation0 = int(np.floor(row_bin)), int(np.floor(column_bin)), int(np.floor(orientation_bin))
                    row_difference, column_difference, orientation_difference = row_bin - row0, column_bin - column0, orientation_bin - orientation0
                    
                    # Distributing weights into the 8 bins
                    for row_index, row_weight in [(row0, 1 - row_difference), (row0 + 1, row_difference)]:
                        if 0 <= row_index < 4:
                            for column_index, column_weight in [(column0, 1 - column_difference), (column0 + 1, column_difference)]:
                                if 0 <= column_index < 4:
                                    for orientation_index, orientation_weight in [(orientation0, 1 - orientation_difference), ((orientation0 + 1), orientation_difference)]:
                                        descriptor_vector[row_index, column_index, orientation_index % 8] += weight * row_weight * column_weight * orientation_weight
                if not valid_window: break
            
            if valid_window:
                vector = descriptor_vector.flatten() # Flattening the grid (4x4 with 8 bins) into 128-dimensional feature vector
                norm = np.linalg.norm(vector) # Using normalization to make vector invariant to light contrast change
                if norm > 1e-6: vector /= norm
                vector[vector > 0.2] = 0.2 # Reducing influence of glare or excess saturation
                norm = np.linalg.norm(vector) # Renormalization
                if norm > 1e-6: vector /= norm
                descriptors.append(vector)
                valid_keypoints.append(keypoint) 
        return valid_keypoints, np.array(descriptors, dtype=np.float32)

# Custom RANSAC implementation (Affine instead of Homography to prevent stretching of the pictures)
def start_custom_ransac(points_source, points_destination, threshold=5.0, max_iterations=2000):
    best_homography = None
    maximum_inliers = 0
    best_mask = None
    n = points_source.shape[0]
    
    # 3 points must be there for affine transformation
    if n < 3: return None, None
    source, destination = np.squeeze(points_source), np.squeeze(points_destination)

    # Finding the best homography and mask
    for _ in range(max_iterations):
        index = np.random.choice(n, 3, replace=False)
        source_triangle, destination_triangle = source[index], destination[index]
        matrix = cv2.getAffineTransform(source_triangle, destination_triangle) # Affine transformation        
        homography = np.vstack([matrix, [0, 0, 1]]) # Conversion from 2x3 Affine to 3x3 Homography format

        # Using the homography model to predict all the points
        source_homogenous = np.hstack((source, np.ones((n, 1))))
        predicted_homogenous = (homography @ source_homogenous.T).T
        predicted_coordinates = predicted_homogenous[:, :2] 
        inliers = np.linalg.norm(destination - predicted_coordinates, axis=1) < threshold # Finding the inliers (distance < threshold)
        count = np.sum(inliers)
        if count > maximum_inliers: # Only keeping the best homography model
            maximum_inliers = count
            best_homography = homography
            best_mask = inliers   
    return best_homography, best_mask

# Image resizing/downscaling method
def resize_image(image, width=None):
    if image is None: return None
    height, image_width = image.shape[:2]
    if width is None or width <= 0 or image_width == 0:
        return image
    scale = width / float(image_width)
    dimension = (int(round(width)), max(1, int(round(height * scale))))
    return cv2.resize(image, dimension, interpolation=cv2.INTER_AREA)

# Sorts the number strings (e.g. 1, 2, 10 instead of 1, 10, 2)
def correct_number_sorting(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# Wrapper which starts the custom SIFT process
def start_custom_sift(image, name="Image"):
    print(f"Starting the detection process for image {name}", end=" ", flush=True)
    sift = CustomSIFT()
    grayscale = image
    if len(image.shape) == 3: grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Generating the features
    difference_of_gaussian = sift.generate_difference_of_gaussian(sift.generate_gaussian_images(grayscale.astype(np.float32), 1.6))
    keypoints_raw = sift.find_keypoints(difference_of_gaussian)
    keypoints, features = sift.compute_descriptor(grayscale, keypoints_raw)
    print(f"{len(keypoints)} keypoints were found")
    return keypoints, features

# Matching keypoints and features
def match_keypoints(keypointsA, keypointsB, featuresA, featuresB, ratio=0.8):
    if len(keypointsA) == 0 or len(keypointsB) == 0: return None # Returning if no keypoints are passed in as arguments
    matches = []
    # Using the euclidean distance matching process
    for i in range(len(featuresA)):
        difference = featuresB - featuresA[i]
        distance = np.linalg.norm(difference, axis=1)
        index_sorted = np.argsort(distance)
        # Best match should be significantly better than second best match - Lowe's ratio test
        if distance[index_sorted[0]] < ratio * distance[index_sorted[1]]:
            matches.append(cv2.DMatch(i, index_sorted[0], distance[index_sorted[0]]))

    # There should be at least 4 matches to proceed further
    if len(matches) > 4:
        # Getting the point coordinates
        pointsA = np.float32([keypointsA[match.queryIdx].pt for match in matches])
        pointsB = np.float32([keypointsB[match.trainIdx].pt for match in matches])
        homography, mask = start_custom_ransac(pointsA, pointsB) # Starting the custom RANSAC for filtering outliers
        if homography is None: return None
        return (matches, homography, mask)
    return None

# Remove black borders from the individual images
def crop_black_borders(panorama):
    if panorama is None or panorama.size == 0: return None # Making sure the parameter is valid
    grayscale = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    threshold = cv2.dilate(threshold, kernel, iterations=2)
    coordinates = cv2.findNonZero(threshold)
    if coordinates is None: return panorama
    x, y, width, height = cv2.boundingRect(coordinates)
    return panorama[y : y + height, x : x + width]

def start_custom_stitching(images):
    print(f"\nBeginning the custom stitching on total {len(images)} images\n")
    
    # Initializing using identity matrix, but will hold the pairwise homographies
    homographies = [np.identity(3)]
    for i in range(len(images) - 1):
        print(f"Aligning the image {i+2} to {i+1}")
        # Starting the feature detection with the custom SIFT implementation
        previous_keypoints, previous_features = start_custom_sift(images[i], f"Image {i+1}")
        current_keypoints, current_features = start_custom_sift(images[i+1], f"Image {i+2}")
        # Matching the features using the custom RANSAC
        result = match_keypoints(current_keypoints, previous_keypoints, current_features, previous_features)
        if result is None: # Only happens if there are not enough feature matches between the two images
            print(f"There were not enough matches between images {i+1} and {i+2} so alignment was not possible")
            return None
        (_, homography_current_to_previous, _) = result
        print(f"Alignment between images {i+2} and {i+1} was successful")
        # Chaining the transformations, effectively maps image n to image n-1 to ... image 1
        homographies.append(homographies[i].dot(homography_current_to_previous))

    # Fixing bowtie distortion effect by recentering the canvas
    print("\nCalculating proper canvas size to recenter")
    middle_index = len(images) // 2 # Calculation of inverse transform for middle image to become the anchor for the panorama
    homography_middle_to_0 = homographies[middle_index]
    homography_0_to_middle = np.linalg.inv(homography_middle_to_0)
    new_homographies = []
    all_corners = []
    
    # Minimizing skew on the far sides by updating the homographies to be relative to center/middle image
    for i, homography in enumerate(homographies):
        new_homography = homography_0_to_middle.dot(homography) 
        new_homographies.append(new_homography)
        height, width = images[i].shape[:2]
        corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, new_homography)) # Applying homography matrix to the 4 corners

    # Finalizing the canvas size
    all_corners = np.concatenate(all_corners, axis=0)
    x_minimum, y_minimum = np.int32(all_corners.min(axis=0).ravel())
    x_maximum, y_maximum = np.int32(all_corners.max(axis=0).ravel())
    out_width, out_height = x_maximum - x_minimum, y_maximum - y_minimum
    print(f"Finalized canvas size is {out_width} x {out_height}")
    print("Putting the panorama together")
    
    # Creating translation matrix to shift any negative coordinates to positive values
    homography_translation = np.array([[1, 0, -x_minimum], [0, 1, -y_minimum], [0, 0, 1]])
    panorama = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    # Center image is added after all other images (it's also the least distorted one)
    draw_order = sorted(range(len(images)), key=lambda x: abs(x - middle_index), reverse=True)
    for i in draw_order: # Iterating through each image in the specified order
        # Applying the final tranformation matrix - alignment, recentering, and translation
        final_homography = homography_translation.dot(new_homographies[i])
        warped = cv2.warpPerspective(images[i], final_homography, (out_width, out_height), flags=cv2.INTER_LINEAR)
        # Generating the mask and eroding dark edges
        grayscale = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) # Grayscale conversion
        _, mask = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY) # Creating the mask
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1) 
        panorama[mask > 0] = warped[mask > 0] # Putting the valid pixels into the canvas for creating the panorama
        
    return crop_black_borders(panorama) # Removing dark black borders and returning the final panorama image


def run_stitching_custom(images, width=None, require_min_images=True):
    """Entry point used by the Flask app to run the custom stitching pipeline."""
    if images is None:
        raise ValueError("Images list cannot be None")

    processed = []
    for image in images:
        if image is None or getattr(image, 'size', 0) == 0:
            continue
        resized = resize_image(image, width=width) if width is not None else image
        processed.append(resized)

    if require_min_images and len(processed) < 4:
        raise ValueError("At least 4 valid images are required for stitching.")

    return start_custom_stitching(processed)

# Using affine transformation rather than Homography to avoid bowtie distortion effect
def start_opencv_stitching(images):
    print(f"\nStarting the OpenCV SIFT/RANSAC stitching on {len(images)} images")
    homographies = [np.identity(3)]
    for i in range(len(images) - 1):
        sift = cv2.SIFT_create()
        previous_keypoint, previous_descriptor = sift.detectAndCompute(images[i], None)
        current_keypoint, current_descriptor = sift.detectAndCompute(images[i + 1], None)
        brute_force_matcher = cv2.BFMatcher()
        matches = brute_force_matcher.knnMatch(current_descriptor, previous_descriptor, k=2)
        valid_matches = []
        for best_match, second_best_match in matches:
            if best_match.distance < 0.75 * second_best_match.distance: valid_matches.append(best_match)
        if len(valid_matches) < 4: # There should be at least 4 matches to continue
            print(f"There were not enough matches between the image {i+1} and {i+2}")
            return None
        source_points = np.float32([current_keypoint[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
        destination_points = np.float32([previous_keypoint[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
        # Affine transformation matrix - rotation, scaling, and translation - no extreme stretching / bowtie effect
        matrix, _ = cv2.estimateAffinePartial2D(source_points, destination_points)
        if matrix is None: # Only happens if OpenCV is unable to calculate the transformation somehow
            print(f"Unable to determine the Affine transformation between the image {i+1} and {i+2}")
            return None
        # Conversion of 2x3 affine tranformation matrix to 3x3 homography matrix
        homography = np.vstack([matrix, [0, 0, 1]])
        homographies.append(homographies[i].dot(homography))

    # Shifting the anchor from image 1 to the middle image
    middle_index = len(images) // 2
    homography_0_to_middle = np.linalg.inv(homographies[middle_index])
    new_homographies = [homography_0_to_middle.dot(homography) for homography in homographies] # Updating all the homographies to be relative to center image
    all_corners = []

    # Minimizing skew on the far sides by updating the homographies to be relative to center/middle image
    for i, homography in enumerate(new_homographies):
        h, w = images[i].shape[:2]
        c = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        all_corners.append(cv2.perspectiveTransform(c, homography))
        
    # Finalizing the canvas size
    all_corners = np.concatenate(all_corners, axis=0)
    x_minimum, y_minimum = np.int32(all_corners.min(axis=0).ravel())
    x_maximum, y_maximum = np.int32(all_corners.max(axis=0).ravel())
    out_width, out_height = x_maximum - x_minimum, y_maximum - y_minimum
    
    # Creating translation matrix to shift any negative coordinates to positive values
    homography_translation = np.array([[1, 0, -x_minimum], [0, 1, -y_minimum], [0, 0, 1]])
    panorama = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    # Center image is added after all other images (it's also the least distorted one)
    draw_order = sorted(range(len(images)), key=lambda x: abs(x - middle_index), reverse=True)
    for i in draw_order: # Iterating through each image in the specified order
        # Applying the final tranformation matrix - alignment, recentering, and translation
        final_homography = homography_translation.dot(new_homographies[i])
        warped = cv2.warpPerspective(images[i], final_homography, (out_width, out_height))
        grayscale = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) # Grayscale conversion
        _, mask = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY) # Creating the mask
        mask = cv2.erode(mask, np.ones((3,3), np.uint8), iterations=1)
        panorama[mask > 0] = warped[mask > 0]  # Putting the valid pixels into the canvas for creating the panorama
        
    return crop_black_borders(panorama) # Removing dark black borders and returning the final panorama image

# Main function
def main():
    relative_image_folder = input("Please enter the path to the image folder :- ").strip()
    image_folder = os.path.abspath(relative_image_folder)
    if not os.path.exists(image_folder):
        print(f"Unable to locate the folder at this path :- {image_folder}")
        return
    
    # Loading images and sorting naturally (1, 2, 10 rather than 1, 10, 2)
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*")), key=lambda s: correct_number_sorting(s))
    original_images = [cv2.imread(path) for path in image_paths if path.lower().endswith(('.jpg', '.png', '.jpeg'))]
    original_images = [image for image in original_images if image is not None]
    if len(original_images) < 4:
        print("At least 4 images need to be loaded")
        return

    images = [resize_image(img, width=WORK_WIDTH) for img in original_images] # Downscaling/resizing for custom SIFT implementation
    custom_sift_result = start_custom_stitching(images) # Run custom SIFT implementation
    opencv_sift_result = start_opencv_stitching(images) # Run OpenCV implementation

    # Saving results without opening any GUI windows
    custom_output = os.path.join(image_folder, "stitched_custom_sift.jpg")
    opencv_output = os.path.join(image_folder, "stitched_opencv_sift.jpg")
    if custom_sift_result is not None:
        cv2.imwrite(custom_output, custom_sift_result)
        print(f"Custom SIFT/RANSAC panorama image saved to {custom_output}")
    else: print("Custom SIFT/RANSAC panorama stitching failed")
    if opencv_sift_result is not None:
        cv2.imwrite(opencv_output, opencv_sift_result)
        print(f"OpenCV SIFT/RANSAC panorama saved to {opencv_output}")
    else: print("OpenCV SIFT/RANSAC panorama stitching failed")

if __name__ == "__main__":
    main()