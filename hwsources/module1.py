"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 1 Assignment - Object Dimension Calculator

The purpose of the script is for calculating the
dimensions of an object by using the perspective
projection equations (Pinhole camera model).

Equation used is W/D = w/f --> W = (w*D)/f, where
 - w is the object's width in terms of pixels
 - f is the camera's focal length in terms of pixels
 - W is the object's width in real life (cm)
 - D is the distance between camera and object (cm)

Process:
 1. User provides reference image path
 2. User selects two points on the reference image to mark object width
 3. User inputs real-world width and distance for the reference object
 4. Program calculates and stores focal length
 5. User provides test image path (must be from same camera)
 6. User selects two points on the test image to mark object width
 7. User inputs distance to the test object
 8. Program calculates and displays real-world width of test object

How to run the program:
 - Run this script: python module1.py
 - Follow the sequential prompts

The dependencies are opencv-python
"""

import cv2
import math
import os

# Focal length calculation
def calculate_focal_length(pixel_width, real_width, distance):
    if real_width == 0:
        raise ValueError("Real world width should be non-zero")
    return (pixel_width * distance) / real_width

# Real world width calculation
def calculate_real_dimension(pixel_width, focal_length, distance):
    if focal_length == 0:
        raise ValueError("Focal length should be non-zero")
    return (pixel_width * distance) / focal_length

# Prompt for getting float values
def prompt_float(prompt_text):
    try:
        return float(input(prompt_text).strip())
    except ValueError:
        print("Invalid input")
        return None

# Prompt for getting the image's file path and loading the image
def prompt_image(label, prompt_text):
    image_path = input(prompt_text).strip()
    image = load_image(image_path)
    if image is None:
        print(f"Unable to load {label.lower()} image :- {image_path}")
    else:
        print(f"{label} image was loaded successfully")
    return image

# Display header at the beginning of the output
def header():
    print("Object Dimension Calculator")
    print("_" * 25)
    print("\nReference & test images should be taken using same camera for good results\n")

# Load the image
def load_image(image_path, max_height=800):
    candidate = os.path.expanduser(image_path)
    if not os.path.exists(candidate):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, image_path)
        if not os.path.exists(candidate):
            return None

    image = cv2.imread(candidate)
    if image is None:
        return None

    height, width = image.shape[:2]
    if height <= max_height:
        return image

    scale = max_height / height
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size)

# User should select two points denoting the dimension of the object
def select_two_points_on_image(image, window_name="Select 2 points"):
    points = []
    display_image = image.copy()

    def mouse_callback(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) == 2: return
        points.append((x, y))
        cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)

        if len(points) == 2:
            cv2.line(display_image, points[0], points[1], (0, 255, 0), 2)
            dist_px = math.dist(points[0], points[1])
            mid_x = (points[0][0] + points[1][0]) // 2
            mid_y = (points[0][1] + points[1][1]) // 2
            cv2.putText(display_image, f"{dist_px:.1f}px", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, display_image)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_image)
    print("\nSelect two points in the image pop up window in order to mark the dimension of the object\n")
    print("Press q to cancel")

    while len(points) < 2:
        if cv2.waitKey(50) & 0xFF == ord("q"):
            cv2.destroyWindow(window_name)
            return None, None

    cv2.waitKey(300)
    cv2.destroyWindow(window_name)
    return tuple(points) if len(points) == 2 else (None, None)

# Calibration step (calculate focal length)
def calibrate_camera():
    print("Calibration (Focal Length)")
    ref_image = prompt_image("Reference", "\nEnter the path to reference image :- ")
    if ref_image is None: return None

    point_a, point_b = select_two_points_on_image(ref_image, "Select two points - Reference image")
    if point_a is None or point_b is None:
        print("Point selection has been cancelled")
        return None

    ref_pixel_width = math.dist(point_a, point_b)
    print(f"\nReference object width in pixels :- {ref_pixel_width:.2f} px")

    ref_real_width = prompt_float("Enter the real world width of object (cm) :- ")
    if ref_real_width is None: return None

    ref_distance = prompt_float("Enter the distance between camera and object (cm) :- ")
    if ref_distance is None: return None

    try:
        focal_length = calculate_focal_length(ref_pixel_width, ref_real_width, ref_distance)
    except ValueError as err:
        print(f"There was an issue :- {err}")
        return None

    print("\n" + "_" * 25)
    print(f"Focal length :- {focal_length:.2f} px")
    print("Focal length has been successfully recorded\n")
    return focal_length

# Object dimension measurement step
def measure_test_object(focal_length):
    print("Object Dimension Measurement")
    test_image = prompt_image("Test", "\nEnter the path for test image :- ")
    if test_image is None: return

    point_a, point_b = select_two_points_on_image(test_image, "Select 2 points - Test image")
    if point_a is None or point_b is None:
        print("Point selection has been cancelled")
        return

    test_pixel_width = math.dist(point_a, point_b)
    print(f"\nTest object width in pixels :- {test_pixel_width:.2f} px")

    test_distance = prompt_float("Enter the distance between camera and the object (cm) :- ")
    if test_distance is None: return

    try:
        test_real_width = calculate_real_dimension(test_pixel_width, focal_length, test_distance)
    except ValueError as err:
        print(f"There was an issue :- {err}")
        return

    print("\nFinal results")
    print(f"Focal Length :- {focal_length:.2f} px")
    print(f"Object width in px :- {test_pixel_width:.2f} px")
    print(f"Object distance :- {test_distance:.2f} cm")
    print(f"\nCalculated real world width :- {test_real_width:.4f} cm")
    print("Thank you for using this program")

# Main function
def main():
    header()
    focal_length = calibrate_camera()
    if focal_length is None:
        return
    measure_test_object(focal_length)

if __name__ == "__main__":
    main()