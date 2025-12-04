"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 3 Assignment Part 2 - Tasks 4-5 (ArUco vs. SAM2 object segmentation)

The purpose of this script is to segment non-rectangular 
objects with OpenCV's ArUco and compare/contrast it with 
the PyTorch SAM2 segmentation model.

Usage:
    1. Run the program - python module3_part2.py
    2. Enter the file path to the image (must contain ArUco markers)
        a. ArUco dictionary 4x4 50 should be used
        b. Markers should be attached on the object boundary
        c. Markers should be placed in order of id beginning with 0, 1, 2... to define the path
    3. Program will perform object segmentation with help of ArUco stickers, 
       and compare/contrast the results after applying SAM2 segmentatin to the same image
    4. If you want, the image can be saved to hwsources/resources/m3/part2

The dependencies are opencv-python, matplotlib, torch, and sam2
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("There was an issue installing PyTorch / SAM2 modules. Will have to skip Task 5")
    print("Please make sure the dependencies have been correctly installed to the proper Python environment")

# Task 4 functionality
def aruco_segmentation(image): 
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale conversion
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # Loading the Aruco dictionary (the type has to be 4x4 50)
    parameters = cv2.aruco.DetectorParameters() # Storing the detection parameters
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters) # Instantiating the Aruco detection object
    corners, ids, rejected = detector.detectMarkers(grayscale) # Getting the markers - corners, marker ids, and rejected candidates
    mask = np.zeros(image.shape[:2], dtype=np.uint8)# Creating empty mask
    sorted_marker_coordinates = [] # Will store the x,y coordinates sorted by marker id

    # Checks to see if at least 3 markers were recognized (3 is minimum required to capture a polygon shape)
    if ids is not None and len(ids) >= 3:
        flattened_ids = ids.flatten() #Flattening the array from [[id_1], [id_2], ...] to [id_1, id_2, ...]
        marker_centers = [] # Will store the center point of each Aruco marker
        # Calculating the center point of given marker by averaging the corner coordinates, then storing it
        for marker_data in corners:
            corner = marker_data[0]
            center_x = int(np.mean(corner[:, 0]))
            center_y = int(np.mean(corner[:, 1]))
            marker_centers.append((center_x, center_y))
        
        # Sort the points based on marker id so that the path between markers has been defined properly
        sorted_pairs = sorted(zip(flattened_ids, marker_centers), key=lambda x: x[0])
        sorted_marker_coordinates = [point for _, point in sorted_pairs]
        # Converting the center points list into a Numpy array and reshaping for OpenCV fillPoly
        points = np.array(sorted_marker_coordinates, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 255) #Filling polygon with white color on the black mask background
        visualization = image.copy() # Image copy to be returned after adding green boundary
        cv2.polylines(visualization, [points], True, (0, 255, 0), 3) # Drawing the green lines
        cv2.aruco.drawDetectedMarkers(visualization, corners, ids) # Putting the markers themselves for reference
        return visualization, mask, np.array(sorted_marker_coordinates)
    else: # Not enough markers were found
        print(f"Only {0 if ids is None else len(ids)} markers were found, at least 3 are required")
        return image, mask, []

# Task 5 functionality
def sam2_segmentation(image, input_points):
    # Returning empty mask if SAM2 was not successfully installed/imported earlier
    if not SAM2_AVAILABLE:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    # SAM2 configuration
    model_checkpoint = "checkpoints/sam2_hiera_large.pt" #SAM2 model checkpoint file
    model_configuration = "sam2_hiera_l.yaml" #SAM2 model configuration file 
    if not os.path.exists(model_checkpoint): # Checking to see if the model checkpoint exists
        print(f"SAM2 model checkpoint file was not found at this location :- {model_checkpoint}")
        print(f"sam2_hiera_large.pt should be downloaded and placed into this location :- {model_checkpoint}")
        return np.zeros(image.shape[:2], dtype=np.uint8) # Returning empty mask if the file was not found

    # If GPU is not available then use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading the SAM2 model, using device :- {device}")

    # Building the SAM2 model using the model configuration and checkpoint files
    sam2_model = build_sam2(model_configuration, model_checkpoint, device=device) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # SAM2 expects RGB format, so image is being converted to that
    predictor = SAM2ImagePredictor(sam2_model) # Predictor object
    predictor.set_image(image_rgb)

    # A bounding box [x_minimum, y_minimum, x_maximum, y_maximum] is created using the Aruco markers
    x_minimum, y_minimum = np.min(input_points, axis=0) #minimum x and y coordinates from the markers' centers
    x_maximum, y_maximum = np.max(input_points, axis=0) #maximum x and y values
    bounding_box = np.array([x_minimum, y_minimum, x_maximum, y_maximum])

    #Running prediction using the bounding box, segmentation masks will be retrieved
    print(f"Running the prediction using the following bounding box :- {bounding_box}")
    masks, _, _ = predictor.predict(box=bounding_box, multimask_output=False)
    sam2_mask = (masks[0] * 255).astype(np.uint8) # Converting the mask from float value to 0-255 range integer values
    return sam2_mask

# Calculation of IoU scores
def calculate_iou_score(mask1, mask2):
    # Conversion to boolean values
    # values > 0 means segmentation, values <= 0 means no segmentation
    mask1_boolean = mask1 > 0
    mask2_boolean = mask2 > 0
    
    # Intersection calculation
    intersection = np.logical_and(mask1_boolean, mask2_boolean).sum()
    # Union calculation
    union = np.logical_or(mask1_boolean, mask2_boolean).sum()
    
    # Checking for division by zero
    if union == 0: return 0.0
    return intersection / union # Return IoU score

# Main function
def main():  
    # Prompting user to enter the image file path
    image_path = input("Please enter the file path to the image (make sure it contains the ArUco markers) :- ").strip()
    if not os.path.exists(image_path): # Making sure that the file exists
        print("File path is invalid, please try again")
        return

    image = cv2.imread(image_path) # OpenCV reads the image in BGR format by default
    if image is None: #In case file is corrupted / unsupported format
        print("Unable to read the file. Could be corrupted / unsupported format")
        return
    
    # Task 4
    print("Starting Task 4 (ArUco)")
    visualization_aruco, mask_aruco, points = aruco_segmentation(image)
    if len(points) == 0: # If there were no Aruco markers then we can exit
        print("No Aruco markers were found in the image, exiting the program")
        return

    # Task 5
    print("Starting Task 5 (SAM2)")
    mask_sam = sam2_segmentation(image, points)

    # IoU score calculation
    if np.max(mask_sam) > 0: #Just checking whether SAM2 created a proper mask (should have some white pixel count)
        intersection_over_union = calculate_iou_score(mask_aruco, mask_sam)
        print(f"\nThe intersection over union score :- {intersection_over_union:.4f}")
    else: # There was no valid mask from SAM2 (Not installed or SAM2 segmentation somehow failed)
        intersection_over_union = 0.0
        print("\nSAM2 was unable to generate a mask, or SAM2 is not installed")

    # Creating the visualizations with matplotlib
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 3, 1) # Aruco visualization
    plt.title("Task 4 - ArUco segmentation")
    plt.imshow(cv2.cvtColor(visualization_aruco, cv2.COLOR_BGR2RGB)) # Conversion from BGR to RGB for OpenCV
    plt.axis('off')

    plt.subplot(1, 3, 2) # SAM2 visualization
    plt.title("Task 5 - SAM2 segmentation")
    visualization_sam2 = image.copy() # Creating proper visualization for SAM2
    visualization_sam2[mask_sam > 0] = [0, 0, 255] #Wherever the mask is white, we are adding a red overlay
    visualization_sam2 = cv2.addWeighted(image, 0.7, visualization_sam2, 0.3, 0) # Blending with original image
    plt.imshow(cv2.cvtColor(visualization_sam2, cv2.COLOR_BGR2RGB))# Conversion from BGR to RGB for OpenCV
    plt.axis('off')

    plt.subplot(1, 3, 3) # Complete visual comparison between Aruco and SAM2
    plt.title(f"ArUco vs. SAM2 - IoU value {intersection_over_union:.2f} comparison")
    comparison_visualization = image.copy().astype(float) # Creating a clean comparison map
    comparison_visualization[:, :, 1] = np.where(mask_aruco > 0, comparison_visualization[:, :, 1] + 100, comparison_visualization[:, :, 1]) # Green color for ArUco
    comparison_visualization[:, :, 2] = np.where(mask_sam > 0, comparison_visualization[:, :, 2] + 100, comparison_visualization[:, :, 2]) # Red color for SAM2
    comparison_visualization = np.clip(comparison_visualization, 0, 255).astype(np.uint8) # Any float values are converted to unsigned int, and no values under 0 or above 255 are allowed
    plt.imshow(cv2.cvtColor(comparison_visualization, cv2.COLOR_BGR2RGB))# Conversion from BGR to RGB for OpenCV
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Saving functionality
    save_plot = input("\nSave the comparison plot as an image? (y/n): ").strip().lower()
    if save_plot == 'y':
        output_folder = "hwsources/resources/m3/part2/"
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.basename(image_path)
        filename = f"comparison_{os.path.splitext(base_name)[0]}.png"
        full_save_path = os.path.join(output_folder, filename)
        plt.savefig(full_save_path)
        print(f"File has been saved to :- {full_save_path}")

if __name__ == "__main__":
    main()