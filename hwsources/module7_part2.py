"""
CSC 8830 Computer Vision
Avyuktkrishna Ramasamy
Dr. Ashwin Ashok
Module 7 Assignment Part 2 - Pose Estimation and Hand Tracking (Mediapipe)

The purpose of this script is to simultaneously track pose 
and hand gestures of people from given input video files. 
Mediapipe has been utilized for this purpose. The script 
processes the video frame by frame in order to :-
 - detection of 30+ body pose landmarks (specific joints or body parts motion)
 - detection of 20+ hand landmarks per hand (handling difference between left/right hands and various actions)
 - visualization of skeletal overlay in real time 
 - exporting the coordinations of all the landmarks to CSV file format

Usage:
    python pose_hand_tracker.py ./data/input_video.mp4 --display
    python pose_hand_tracker.py ./data/input_video.mp4 --no-display

Major dependency is Mediapipe
"""

from __future__ import annotations
import cv2
import argparse
import csv
import sys
import mediapipe as mp
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# Keeps MediaPipe solution handles handy so we do not keep re-importing
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Calls a MediaPipe helper when it exists, otherwise uses our own fallback spec
def _call_with_fallback(module, attr: str, fallback):
    fn = getattr(module, attr, None)
    return fn() if callable(fn) else fallback


# Parses CLI arguments so users can pick a video and toggle the preview window
def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose and hand landmark extraction with optional visualization.")
    parser.add_argument("video_path", type=Path, help="Path to the input video file to analyze.")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True, help="Show a real-time window with landmarks.")
    return parser.parse_args(argv)


# Builds the CSV header covering every pose and hand landmark coordinate
def _build_fieldnames() -> List[str]:
    fieldnames: List[str] = ["frame_index", "time_sec"]

    for landmark in mp_pose.PoseLandmark:
        name = landmark.name.lower()
        fieldnames.extend([
            f"pose_{name}_x",
            f"pose_{name}_y",
            f"pose_{name}_z",
            f"pose_{name}_visibility",
        ])

    for side in ("left", "right"):
        for landmark in mp_hands.HandLandmark:
            name = landmark.name.lower()
            fieldnames.extend([
                f"{side}_hand_{name}_x",
                f"{side}_hand_{name}_y",
                f"{side}_hand_{name}_z",
            ])

    return fieldnames


# Sets up an empty dictionary for each frame before we populate it
def _initialize_frame_record(fieldnames: List[str]) -> Dict[str, float]:
    return {field: None for field in fieldnames}


# Names the CSV output based on the source video filename
def _derive_output_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_pose_hand.csv")


# Grabs MediaPipe's default drawing styles or simple fallbacks when missing
def _drawing_specs():
    return (
        _call_with_fallback(mp_drawing_styles, "get_default_pose_landmarks_style", mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)),
        _call_with_fallback(mp_drawing_styles, "get_default_pose_connection_style", mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)),
        _call_with_fallback(mp_drawing_styles, "get_default_hand_landmarks_style", mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)),
        _call_with_fallback(mp_drawing_styles, "get_default_hand_connections_style", mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)),
    )


# Draws pose and hand skeletons over the current frame when visualization is requested
def _annotate(frame, pose_results, hands_results, specs):
    pose_landmark_spec, pose_connection_spec, hand_landmark_spec, hand_connection_spec = specs
    annotated = frame.copy()
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=pose_landmark_spec,
            connection_drawing_spec=pose_connection_spec,
        )
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_spec,
                connection_drawing_spec=hand_connection_spec,
            )
    return annotated


# Picks whether this detected hand should be treated as left or right for CSV labels
def _label_for_hand(idx: int, handedness: Dict[int, str], side_usage: Dict[str, bool]) -> str:
    side = handedness.get(idx, "")
    if side in ("left", "right"):
        side_usage[side] = True
        return side
    for candidate in ("left", "right"):
        if not side_usage.get(candidate):
            side_usage[candidate] = True
            return candidate
    return f"hand_{idx}"


# Spins up the annotated video writer lazily once we know frame dimensions
def _ensure_writer(writer, path: Optional[Path], fps: float, frame) -> Optional[cv2.VideoWriter]:
    if writer or not path:
        return writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        writer.release()
        return None
    return writer


# Coordinates video decoding, MediaPipe inference, CSV logging, and optional visualization
def _run_estimation(video_path: Path, display: bool, annotated_output_path: Optional[Path] = None) -> Dict[str, object]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps_safe = fps if fps > 0 else 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    annotated_path = Path(annotated_output_path) if annotated_output_path else None
    if annotated_path:
        annotated_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _build_fieldnames()
    records: List[Dict[str, float]] = []
    video_writer = None
    drawing_specs = _drawing_specs()

    with mp_pose.Pose(model_complexity=1, enable_segmentation=False) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    ) as hands:
        frame_index = 0
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if frame_width <= 0 or frame_height <= 0:
                    frame_height, frame_width = frame.shape[:2]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                pose_results = pose.process(frame_rgb)
                hands_results = hands.process(frame_rgb)
                frame_rgb.flags.writeable = True

                record = _initialize_frame_record(fieldnames)
                record["frame_index"] = frame_index
                record["time_sec"] = frame_index / fps if fps > 0 else None

                if pose_results.pose_landmarks:
                    for landmark_enum, landmark in zip(mp_pose.PoseLandmark, pose_results.pose_landmarks.landmark):
                        name = landmark_enum.name.lower()
                        record[f"pose_{name}_x"] = landmark.x
                        record[f"pose_{name}_y"] = landmark.y
                        record[f"pose_{name}_z"] = landmark.z
                        record[f"pose_{name}_visibility"] = landmark.visibility

                handedness = {
                    idx: info.classification[0].label.lower()
                    for idx, info in enumerate(hands_results.multi_handedness or [])
                }
                side_usage: Dict[str, bool] = {"left": False, "right": False}

                if hands_results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                        side = _label_for_hand(idx, handedness, side_usage)
                        for landmark_enum, landmark in zip(mp_hands.HandLandmark, hand_landmarks.landmark):
                            name = landmark_enum.name.lower()
                            record[f"{side}_hand_{name}_x"] = landmark.x
                            record[f"{side}_hand_{name}_y"] = landmark.y
                            record[f"{side}_hand_{name}_z"] = landmark.z

                records.append(record)

                if display or annotated_path is not None:
                    annotated_frame = _annotate(frame, pose_results, hands_results, drawing_specs)
                    video_writer = _ensure_writer(video_writer, annotated_path, fps_safe, annotated_frame)
                    if video_writer is not None:
                        video_writer.write(annotated_frame)
                    if display:
                        cv2.putText(annotated_frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow("Pose and Hand Tracking", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                frame_index += 1
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()

    output_path = _derive_output_path(video_path)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    duration = frame_index / fps if fps > 0 else None
    return {
        "csv_path": output_path,
        "record_count": len(records),
        "frame_count": frame_index,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "duration_seconds": duration,
        "annotated_video_path": annotated_path if annotated_path and annotated_path.exists() else None,
    }


# Handles CLI execution and prints out where artifacts were saved
def main(argv: Iterable[str]) -> None:
    args = _parse_args(argv)
    try:
        result = _run_estimation(args.video_path, args.display)
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc))

    output_path = result.get("csv_path")
    if output_path:
        print(f"Pose and hand landmarks saved to {output_path}")
    annotated_path = result.get("annotated_video_path")
    if annotated_path:
        print(f"Annotated video exported to {annotated_path}")


# Exposes the tracker as a simple function for other modules to call
def run_pose_and_hand_tracking(video_path: Path, *, display: bool = False, annotated_output_path: Optional[Path] = None) -> Dict[str, object]:
    return _run_estimation(video_path, display=display, annotated_output_path=annotated_output_path)


if __name__ == "__main__":
    main(sys.argv[1:])
