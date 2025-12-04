"""
CSC 8830 Computer Vision
Avyuktkrishna Ramasamy
Dr. Ashwin Ashok
Module 5-6 Assignment Part 1 - Real time Object Tracker using ArUco Markers

The purpose of this script is to track movting objects 
real time using ArUco markers. An ArUco marker should 
first be placed on the object, then the program will 
scan the input stream to identify  the specific ArUco 
dictionary used - so there is no need for manually 
specifying the marker itself. 

Usage:
    python script.py path/to/video.mp4 --output result.mp4
    python script.py camera --show-window --dict 4x4_50
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

# Tracks ArUco markers in prerecorded videos or a live webcam feed
def choose_aruco_dictionary(dict_name: Optional[str] = None):
    # Maps user-friendly string names to OpenCV internal constants
    name_map = {
        None: cv2.aruco.DICT_ARUCO_ORIGINAL,
        "4x4_50": cv2.aruco.DICT_4X4_50,
        "5x5_100": cv2.aruco.DICT_5X5_100,
        "6x6_250": cv2.aruco.DICT_6X6_250,
        "original": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "april_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    # Return the requested dictionary or default to Original if not found
    return cv2.aruco.getPredefinedDictionary(name_map.get(dict_name, name_map[None]))


# Finds marker corners and IDs in the current frame
def detect_markers(frame: np.ndarray, aruco_dict, aruco_params) -> Tuple[np.ndarray, np.ndarray]:
    # ArUco detection requires a grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    return corners, ids


# Turns marker corners into a padded bounding box and its center point
def rect_from_corners(corners: np.ndarray, padding: float = 1.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    # Flatten corner array to shape (N, 2) for minAreaRect
    rect = cv2.minAreaRect(corners.reshape(-1, 2))
    
    # Apply padding if requested (scales width and height)
    if padding != 1.0:
        (cx, cy), (w, h), angle = rect
        rect = ((cx, cy), (w * padding, h * padding), angle)
    
    # Convert rotated rectangle back to 4 integer corner points
    box = cv2.boxPoints(rect).astype(int)
    center = (int(rect[0][0]), int(rect[0][1]))
    return box, center


# Draws the marker outline and ID on top of the frame
def draw_marker_box(frame: np.ndarray, box: np.ndarray, marker_id: int) -> None:
    # Draw the polygon defined by 'box' points
    cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 0), thickness=3)
    
    # Calculate text position near the top-left corner
    tl = tuple(box[0])
    cv2.putText(frame, f"ID:{marker_id}", (tl[0] + 4, tl[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# Opens the requested video file or defaults to the webcam
def open_video_capture(path: str):
    # If path is 'camera', pass 0 (index for default webcam), else pass file path
    return cv2.VideoCapture(0 if path == "camera" else path)


# Makes sure the provided path exists unless it is the webcam keyword
def validate_input_path(path: Optional[str]) -> Optional[str]:
    if not path or path == "camera":
        return path
    p = Path(path)
    return str(p) if p.is_file() else None


# Builds detector parameters regardless of the OpenCV version
def build_detector_params():
    # Handles API changes between OpenCV < 4.7 and OpenCV >= 4.7
    return cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, "DetectorParameters_create") else cv2.aruco.DetectorParameters()


# Opens a video writer with reasonable codec fallbacks
def ensure_writer(output_path: Optional[str], fps: float, w: int, h: int) -> Optional[cv2.VideoWriter]:
    if not output_path:
        return None

    resolved = str(output_path)
    suffix = Path(resolved).suffix.lower()
    
    # Define a priority list of codecs based on file extension
    codec_candidates = {
        ".mp4": ["avc1", "H264", "mp4v"],
        ".mov": ["avc1", "H264", "mp4v"],
        ".m4v": ["avc1", "H264", "mp4v"],
        ".avi": ["XVID", "mp4v"],
    }.get(suffix, ["mp4v"])

    # Try each codec until one successfully opens the writer
    for code in codec_candidates:
        writer = cv2.VideoWriter(resolved, cv2.VideoWriter_fourcc(*code), fps, (w, h))
        if writer.isOpened():
            print(f"[module5_6_part1] Using codec {code} for output {resolved}")
            return writer
        writer.release()
        print(f"[module5_6_part1] Failed to open VideoWriter with codec {code}, trying next candidate…")

    print(f"[module5_6_part1] Could not initialize a VideoWriter for {resolved}.")
    return None


# Streams frames, annotates detections, and writes the result if requested
def process_video(
    input_video: str,
    output_video: Optional[str] = None,
    aruco_dict_name: Optional[str] = None,
    padding: float = 1.0,
    show_window: bool = False,
):
    cap = open_video_capture(input_video)
    if not cap or not cap.isOpened():
        print(f"Failed to open input {input_video}")
        return

    # Determine which dictionary to use (manual or auto-detect)
    aruco_dict = None if aruco_dict_name == "auto" else choose_aruco_dictionary(aruco_dict_name)
    aruco_params = build_detector_params()

    if aruco_dict is None:
        print("Auto-detecting ArUco dictionary...")
        aruco_dict = auto_detect_dictionary(input_video, max_frames=500) or choose_aruco_dictionary()

    # Retrieve video metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = ensure_writer(output_video, fps, w, h)
    if output_video and writer is None:
        raise RuntimeError(f"Unable to initialize video writer for {output_video}. Please install H.264/MP4 codecs.")

    print(f"Processing {'camera' if input_video == 'camera' else input_video}: {w}x{h} @ {fps:.2f} FPS")
    if writer:
        print(f"Writing output to {output_video}")

    # Persistence variables: keep track of the last known location
    last_center = None
    last_box = None
    frames_no_detection = 0
    max_keep = 6  # How many frames to "hold" the box if detection is lost

    window_name = "Module 5/6 Tracker"

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video stream

        corners, ids = detect_markers(frame, aruco_dict, aruco_params)
        
        if ids is not None and len(ids) > 0:
            # Marker detected: update state and draw
            for marker_id, marker_corners in zip(ids.flatten(), corners):
                box, center = rect_from_corners(marker_corners, padding=padding)
                last_center, last_box = center, box
                frames_no_detection = 0
                draw_marker_box(frame, box, int(marker_id))
        else:
            # No marker: check if we should draw the "ghost" box from previous frames
            frames_no_detection += 1
            if frames_no_detection <= max_keep and last_box is not None:
                # Draw a thinner line to indicate this is a cached detection
                cv2.polylines(frame, [last_box], isClosed=True, color=(0, 255, 0), thickness=2)
                if last_center:
                    cv2.putText(frame, "(last)", (last_center[0] - 40, last_center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if writer:
            writer.write(frame)

        if show_window:
            cv2.imshow(window_name, frame)
            # Check for ESC key (ASCII 27)
            if cv2.waitKey(1) & 0xFF == 27: 
                break

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyWindow(window_name)


# Handles CLI parsing and kicks off processing
def main(argv=None):
    parser = argparse.ArgumentParser(description="ArUco marker tracker (no AI/ML)")
    parser.add_argument("input", nargs="?", help="Path to input video file (or 'camera' for webcam)")
    parser.add_argument("--output", "-o", help="Path to write annotated output video (optional)")
    parser.add_argument("--dict", help="Aruco dictionary name (optional) e.g. 4x4_50, 5x5_100")
    parser.add_argument("--padding", type=float, default=1.0, help="Padding factor to enlarge bounding box around the marker (default 1.0)")
    parser.add_argument("--show-window", action="store_true", help="Display annotated frames in a preview window while processing")
    args = parser.parse_args(argv)

    # Set default sample file if no input provided
    sample_default = Path("hwsources/resources/m5_6/aruco-marker.mp4")
    input_path = args.input or (str(sample_default) if sample_default.is_file() else None)
    
    input_path = validate_input_path(input_path)
    if not input_path:
        print("Invalid or missing input path. Please pass a valid video file path or 'camera'.")
        sys.exit(1)

    # Generate an output path automatically if not provided
    if args.output:
        output_path = args.output
    else:
        in_path = Path(input_path)
        # Preserve original extension unless it's complex, default to mp4
        suffix = ".mp4" if in_path.suffix == ".mp4" else in_path.suffix
        output_path = str(in_path.with_name(f"{in_path.stem}-tracked{suffix}"))

    process_video(
        input_path,
        output_path,
        args.dict or None,
        padding=args.padding,
        show_window=args.show_window,
    )


# Tries each known dictionary until one actually detects a marker
def auto_detect_dictionary(video_path: str, max_frames: int = 500):
    try:
        # List of most common dictionaries to test
        for name in ["original", "4x4_50", "5x5_100", "6x6_250", "7x7_50", "april_36h11"]:
            dictionary = choose_aruco_dictionary(name)
            cap = open_video_capture(video_path)
            
            if not cap or not cap.isOpened():
                continue
                
            params = build_detector_params()
            
            # Scan the first 'max_frames' looking for a hit
            for _ in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                _, ids = detect_markers(frame, dictionary, params)
                if ids is not None and len(ids) > 0:
                    cap.release()
                    print(f"Auto-detected dictionary: {name}")
                    return dictionary
            cap.release()
    except Exception:
        pass
    return None


if __name__ == "__main__":
    main()