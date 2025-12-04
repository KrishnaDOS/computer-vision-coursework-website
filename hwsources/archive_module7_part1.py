"""Module 7 Part 1 - Calibrated stereo based object dimension calculator.

This script extends the Module 1 single-image measurement workflow by using
calibrated stereo imagery to recover 3D coordinates for any pixel pair the user
marks. Each measurement is the Euclidean distance between two clicked points in
3D, allowing the user to gather width + length for rectangles, diameters for
circular objects, or every visible edge of an arbitrary polygon.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class StereoWorkspace:
	rectified_left: np.ndarray
	rectified_right: np.ndarray
	disparity: np.ndarray
	points_3d: np.ndarray
	q_matrix: np.ndarray


WINDOW_NAME = "Stereo Measurement"


def header() -> None:
	print("Calibrated Stereo Object Dimension Calculator")
	print("_" * 45)
	print(
		"\nSteps: provide stereo calibration, load left/right images, choose the"
		" object type, then draw as many line segments as needed to collect"
		" real-world dimensions (in cm). Press 'r' to reset a measurement or"
		" 'q' to abort.\n"
	)


def resolve_existing_path(path_str: str) -> Optional[str]:
	candidate = os.path.expanduser(path_str)
	if os.path.exists(candidate):
		return candidate
	script_dir = os.path.dirname(os.path.abspath(__file__))
	fallback = os.path.join(script_dir, path_str)
	return fallback if os.path.exists(fallback) else None


def prompt_path(prompt_text: str) -> Optional[str]:
	while True:
		path_str = input(prompt_text).strip()
		if not path_str:
			print("Input cannot be empty. Enter 'q' to exit.")
			continue
		if path_str.lower() == "q":
			return None
		resolved = resolve_existing_path(path_str)
		if resolved:
			return resolved
		print(f"Path not found: {path_str}")


def prompt_float(prompt_text: str, default: Optional[float] = None) -> Optional[float]:
	while True:
		raw = input(prompt_text).strip()
		if not raw and default is not None:
			return default
		if raw.lower() == "q":
			return None
		try:
			return float(raw)
		except ValueError:
			print("Please enter a valid number or 'q' to exit.")


def prompt_int(prompt_text: str, minimum: int = 1) -> Optional[int]:
	while True:
		raw = input(prompt_text).strip()
		if raw.lower() == "q":
			return None
		try:
			value = int(raw)
		except ValueError:
			print("Enter a whole number or 'q' to exit.")
			continue
		if value < minimum:
			print(f"Value should be >= {minimum}.")
			continue
		return value


def prompt_choice(prompt_text: str, options: Sequence[str]) -> Optional[str]:
	lowered = {opt.lower(): opt for opt in options}
	while True:
		raw = input(prompt_text).strip().lower()
		if raw == "q":
			return None
		if raw in lowered:
			return lowered[raw]
		print(f"Enter one of: {', '.join(options)} (or 'q' to exit)")


def load_image(path: str, max_height: int = 1200) -> Optional[np.ndarray]:
	image = cv2.imread(path)
	if image is None:
		return None
	height, width = image.shape[:2]
	if height <= max_height:
		return image
	scale = max_height / float(height)
	new_size = (int(width * scale), int(height * scale))
	return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def load_stereo_calibration(path: str) -> Dict[str, np.ndarray]:
	ext = os.path.splitext(path)[1].lower()
	if ext == ".npz":
		data = np.load(path)
		return {
			"camera_matrix_left": _fetch(data, ["camera_matrix_left", "mtx_left", "cameraMatrix1", "K_left"]),
			"camera_matrix_right": _fetch(data, ["camera_matrix_right", "mtx_right", "cameraMatrix2", "K_right"]),
			"dist_coeffs_left": _fetch(data, ["dist_coeffs_left", "dist_left", "distCoeffs1"]),
			"dist_coeffs_right": _fetch(data, ["dist_coeffs_right", "dist_right", "distCoeffs2"]),
			"rotation": _fetch(data, ["rotation", "R", "R1"]),
			"translation": _fetch(data, ["translation", "T", "T1"]),
		}
	if ext in {".yml", ".yaml", ".xml"}:
		return _load_from_filestorage(path)
	raise ValueError("Unsupported calibration file. Use .npz, .yml, .yaml, or .xml")


def _fetch(data: Dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
	for key in keys:
		if key in data:
			return data[key]
	raise KeyError(f"None of the expected keys {keys} found in calibration file")


def _load_from_filestorage(path: str) -> Dict[str, np.ndarray]:
	storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
	if not storage.isOpened():
		raise ValueError(f"Unable to open calibration file: {path}")
	try:
		return {
			"camera_matrix_left": _read_node(storage, ["camera_matrix_left", "mtx_left", "K1"]),
			"camera_matrix_right": _read_node(storage, ["camera_matrix_right", "mtx_right", "K2"]),
			"dist_coeffs_left": _read_node(storage, ["dist_coeffs_left", "dist_left", "D1"]),
			"dist_coeffs_right": _read_node(storage, ["dist_coeffs_right", "dist_right", "D2"]),
			"rotation": _read_node(storage, ["rotation", "R", "R1"]),
			"translation": _read_node(storage, ["translation", "T", "T1"]),
		}
	finally:
		storage.release()


def _read_node(storage: cv2.FileStorage, names: Sequence[str]) -> np.ndarray:
	for name in names:
		node = storage.getNode(name)
		if not node.empty():
			return node.mat()
	raise KeyError(f"Missing calibration node(s): {names}")


def prepare_workspace(calibration: Dict[str, np.ndarray], left: np.ndarray, right: np.ndarray) -> StereoWorkspace:
	if left.shape != right.shape:
		raise ValueError("Left and right images must share the same resolution")

	camera_matrix_left = calibration["camera_matrix_left"].astype(np.float64)
	camera_matrix_right = calibration["camera_matrix_right"].astype(np.float64)
	dist_left = calibration["dist_coeffs_left"].astype(np.float64)
	dist_right = calibration["dist_coeffs_right"].astype(np.float64)
	rotation = calibration["rotation"].astype(np.float64)
	translation = calibration["translation"].astype(np.float64)

	image_size = (left.shape[1], left.shape[0])
	stereo_rectify = cv2.stereoRectify(
		camera_matrix_left,
		dist_left,
		camera_matrix_right,
		dist_right,
		image_size,
		rotation,
		translation,
		flags=cv2.CALIB_ZERO_DISPARITY,
		alpha=0,
	)
	r1, r2, p1, p2, q, _, _ = stereo_rectify

	map1_left = cv2.initUndistortRectifyMap(
		camera_matrix_left, dist_left, r1, p1, image_size, cv2.CV_32FC1
	)
	map1_right = cv2.initUndistortRectifyMap(
		camera_matrix_right, dist_right, r2, p2, image_size, cv2.CV_32FC1
	)

	rectified_left = cv2.remap(left, map1_left[0], map1_left[1], interpolation=cv2.INTER_LINEAR)
	rectified_right = cv2.remap(right, map1_right[0], map1_right[1], interpolation=cv2.INTER_LINEAR)

	disparity = compute_disparity(rectified_left, rectified_right)
	points_3d = reproject_to_3d(disparity, q)
	return StereoWorkspace(rectified_left, rectified_right, disparity, points_3d, q)


def compute_disparity(rectified_left: np.ndarray, rectified_right: np.ndarray) -> np.ndarray:
	gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
	gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
	width = gray_left.shape[1]
	num_disparities = max(64, math.ceil(width / 64) * 16)
	block_size = 5
	matcher = cv2.StereoSGBM_create(
		minDisparity=0,
		numDisparities=num_disparities,
		blockSize=block_size,
		P1=8 * block_size * block_size,
		P2=32 * block_size * block_size,
		disp12MaxDiff=1,
		uniquenessRatio=5,
		speckleWindowSize=50,
		speckleRange=1,
		preFilterCap=31,
		mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
	)
	disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
	disparity = cv2.medianBlur(disparity, 5)
	disparity[disparity <= 0.0] = np.nan
	return disparity


def reproject_to_3d(disparity: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
	safe_disparity = np.nan_to_num(disparity, nan=0.0)
	points = cv2.reprojectImageTo3D(safe_disparity, q_matrix)
	invalid = ~np.isfinite(disparity)
	points[invalid] = np.nan
	return points.astype(np.float32)


def select_measurement_type() -> Optional[str]:
	print("\nMeasurement types available: rectangle | circle | polygon | custom")
	return prompt_choice("Enter measurement type :- ", ["rectangle", "circle", "polygon", "custom"])


def build_measurement_plan(kind: str) -> Optional[List[str]]:
	if kind == "rectangle":
		return ["Width", "Length"]
	if kind == "circle":
		return ["Diameter"]
	if kind == "polygon":
		count = prompt_int("Enter number of polygon edges to measure :- ", minimum=1)
		if count is None:
			return None
		return [f"Edge {idx + 1}" for idx in range(count)]
	if kind == "custom":
		count = prompt_int("Enter number of initial segments to measure :- ", minimum=1)
		if count is None:
			return None
		return [f"Segment {idx + 1}" for idx in range(count)]
	return None


def sample_world_point(points_3d: np.ndarray, point: Tuple[int, int], cm_per_unit: float) -> Optional[np.ndarray]:
	h, w = points_3d.shape[:2]
	x, y = point
	if not (0 <= x < w and 0 <= y < h):
		return None
	x0 = int(np.floor(x))
	y0 = int(np.floor(y))
	x1 = min(x0 + 1, w - 1)
	y1 = min(y0 + 1, h - 1)
	dx = x - x0
	dy = y - y0

	def get(ix: int, iy: int) -> np.ndarray:
		return points_3d[iy, ix]

	q00 = get(x0, y0)
	q01 = get(x0, y1)
	q10 = get(x1, y0)
	q11 = get(x1, y1)
	if any(np.any(np.isnan(q)) for q in (q00, q01, q10, q11)):
		return None

	top = q00 * (1 - dx) + q10 * dx
	bottom = q01 * (1 - dx) + q11 * dx
	blended = top * (1 - dy) + bottom * dy
	return blended.astype(np.float32) * cm_per_unit


def select_two_points(image: np.ndarray, label: str) -> Optional[List[Tuple[int, int]]]:
	display = image.copy()
	instruction = (
		f"{label}: click two points. Press 'r' to reset this label, 'q' to abort measurement."
	)
	_overlay_text(display, instruction)
	points: List[Tuple[int, int]] = []

	def mouse_callback(event, x, y, _flags, _param):
		if event != cv2.EVENT_LBUTTONDOWN or len(points) == 2:
			return
		points.append((x, y))
		cv2.circle(display, (x, y), 6, (0, 255, 255), -1)
		if len(points) == 2:
			cv2.line(display, points[0], points[1], (0, 255, 0), 2)
			_overlay_text(display, instruction)
		cv2.imshow(WINDOW_NAME, display)

	cv2.namedWindow(WINDOW_NAME)
	cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
	cv2.imshow(WINDOW_NAME, display)

	while True:
		key = cv2.waitKey(50) & 0xFF
		if key == ord("q"):
			cv2.destroyWindow(WINDOW_NAME)
			return None
		if key == ord("r"):
			points.clear()
			display[:] = image
			_overlay_text(display, instruction)
			cv2.imshow(WINDOW_NAME, display)
		if len(points) == 2:
			cv2.waitKey(300)
			cv2.destroyWindow(WINDOW_NAME)
			return points


def measure_segments(
	workspace: StereoWorkspace,
	cm_per_unit: float,
	labels: Sequence[str],
	allow_extra: bool = True,
) -> List[Dict[str, object]]:
	results: List[Dict[str, object]] = []
	for label in labels:
		result = _measure_single_segment(workspace, cm_per_unit, label)
		if result is None:
			return results
		results.append(result)

	while allow_extra:
		cont = input("Add another measurement? (y/n) :- ").strip().lower()
		if cont not in {"y", "yes"}:
			break
		label = input("Enter label for this segment :- ").strip() or f"Segment {len(results) + 1}"
		result = _measure_single_segment(workspace, cm_per_unit, label)
		if result is None:
			break
		results.append(result)
	return results


def _measure_single_segment(
	workspace: StereoWorkspace,
	cm_per_unit: float,
	label: str,
) -> Optional[Dict[str, object]]:
	while True:
		points = select_two_points(workspace.rectified_left, label)
		if points is None:
			print("Measurement cancelled by user.")
			return None
		world_pts = [sample_world_point(workspace.points_3d, pt, cm_per_unit) for pt in points]
		if any(wp is None for wp in world_pts):
			print("Depth could not be estimated for one of the selected pixels. Try again.")
			continue
		length_cm = float(np.linalg.norm(world_pts[0] - world_pts[1]))
		print(f"{label}: {length_cm:.3f} cm")
		_show_measurement_preview(workspace.rectified_left, points, length_cm, label)
		return {"label": label, "pixels": points, "length_cm": length_cm}


def _show_measurement_preview(
	image: np.ndarray,
	points: Sequence[Tuple[int, int]],
	length_cm: float,
	label: str,
) -> None:
	preview = image.copy()
	cv2.line(preview, points[0], points[1], (0, 0, 255), 2)
	mid_x = int((points[0][0] + points[1][0]) / 2)
	mid_y = int((points[0][1] + points[1][1]) / 2)
	cv2.putText(
		preview,
		f"{label}: {length_cm:.2f} cm",
		(mid_x, max(25, mid_y - 10)),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.6,
		(255, 255, 255),
		2,
	)
	window = f"Result - {label}"
	cv2.imshow(window, preview)
	cv2.waitKey(600)
	cv2.destroyWindow(window)


def _overlay_text(image: np.ndarray, text: str) -> None:
	lines = textwrap(text, width=50)
	padding = 6
	font = cv2.FONT_HERSHEY_SIMPLEX
	y = 25
	for line in lines:
		(text_width, _), baseline = cv2.getTextSize(line, font, 0.5, 1)
		cv2.rectangle(
			image,
			(5, y - 15),
			(10 + text_width, y + baseline),
			(0, 0, 0),
			-1,
		)
		cv2.putText(image, line, (10, y), font, 0.5, (255, 255, 255), 1)
		y += baseline + padding + 10


def textwrap(text: str, width: int) -> List[str]:
	words = text.split()
	if not words:
		return []
	lines: List[str] = []
	current = words[0]
	for word in words[1:]:
		if len(current) + 1 + len(word) <= width:
			current += " " + word
		else:
			lines.append(current)
			current = word
	lines.append(current)
	return lines


def summarize_results(results: Sequence[Dict[str, object]]) -> None:
	if not results:
		print("No measurements collected.")
		return
	print("\nFinal Measurements (cm):")
	print("-" * 32)
	for entry in results:
		print(f"{entry['label']:<15} : {entry['length_cm']:.3f} cm")


def main() -> None:
	header()
	calib_path = prompt_path("Enter stereo calibration file path :- ")
	if calib_path is None:
		return
	try:
		calibration = load_stereo_calibration(calib_path)
	except (ValueError, KeyError) as exc:
		print(f"Unable to load calibration: {exc}")
		return

	left_path = prompt_path("Enter LEFT image path :- ")
	if left_path is None:
		return
	right_path = prompt_path("Enter RIGHT image path :- ")
	if right_path is None:
		return

	left_image = load_image(left_path)
	right_image = load_image(right_path)
	if left_image is None or right_image is None:
		print("Unable to load one of the stereo images.")
		return

	scale = prompt_float(
		"Enter conversion factor (cm per world-unit. Default=1) :- ",
		default=1.0,
	)
	if scale is None:
		return

	measurement_type = select_measurement_type()
	if measurement_type is None:
		return
	plan = build_measurement_plan(measurement_type)
	if not plan:
		print("No measurement plan created.")
		return

	try:
		workspace = prepare_workspace(calibration, left_image, right_image)
	except ValueError as exc:
		print(f"Unable to prepare stereo workspace: {exc}")
		return

	print("\nRectified images ready. Beginning measurements...\n")
	results = measure_segments(workspace, scale, plan, allow_extra=True)
	summarize_results(results)
	print("\nRemember: You need a valid stereo calibration (.npz/.yml) and true left/right"
		  " images captured simultaneously. Provide those assets if they are missing"
		  " from the resources folder before running this script.")


if __name__ == "__main__":
	main()
