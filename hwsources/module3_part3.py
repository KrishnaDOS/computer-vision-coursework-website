"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 3 Assignment Part 3 - SAM2 segmentation with Aruco markers

The purpose of this script is to use SAM2 segmentation for 
boundary detection with Aruco stickers. SAM2 stands for 
Segment Anything Model 2 from Meta. Comparative analysis is 
performed between using OpenCV's Aruco marker detection libraries and SAM2

Some of the key features include :-
1) automatic conversion of Aruco boundaries into SAM2 prompts
2) IoU scores are calculated (intersection over union)
3) Visualizations have been generated (baseline vs. SAM2).

Usage:
    python module3_part3.py ./data/images --model-config ./sam2_configs/sam2_hiera_l.yaml --checkpoint ./checkpoints/sam2_hiera_large.pt  --bbox-scale 0.1
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

import cv2
import numpy as np
import torch
import module3_part2 as aruco_module


LOGGER = logging.getLogger("module3_part3")


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Sam2Runtime:
	# Keeps the predictor and device in one bundle.
	predictor: Any
	device: str


def _build_argument_parser() -> argparse.ArgumentParser:
	# Builds the command line options people can tweak.
	parser = argparse.ArgumentParser(
		description="Compare ArUco segmentation with SAM2 outputs for a folder of images.",
	)
	parser.add_argument(
		"image_folder",
		type=Path,
		help="Folder containing the evaluation images (non-recursive).",
	)
	parser.add_argument(
		"--suffix",
		default="_sam2",
		help="Suffix appended to the visualization output files (default: _sam2).",
	)
	parser.add_argument(
		"--aruco-iterations",
		type=int,
		default=5,
		help="Number of GrabCut iterations for the ArUco baseline (default: 5).",
	)
	parser.add_argument(
		"--aruco-hull-expansion",
		type=float,
		default=0.15,
		help="Fractional hull expansion reused from Part 2 (default: 0.15).",
	)
	parser.add_argument(
		"--bbox-scale",
		type=float,
		default=0.05,
		help="Extra padding applied to the SAM2 bounding box prompt (default: 0.05).",
	)
	parser.add_argument(
		"--prompt-points",
		type=int,
		default=4,
		help="Number of positive point prompts sampled from the hull (default: 4, 0 to disable).",
	)
	parser.add_argument(
		"--model-config",
		type=Path,
		required=True,
		help="Absolute or relative path to the SAM2 YAML config downloaded locally.",
	)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		required=True,
		help="Absolute or relative path to the SAM2 model weights (.pt) stored locally.",
	)
	parser.add_argument(
		"--device",
		default="auto",
		help="Torch device to use (cpu/cuda/mps). The default 'auto' picks CUDA when available.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Enable debug logging.",
	)
	return parser


def _setup_logging(verbose: bool) -> None:
	# Chooses a simple logging level for the run.
	logging.basicConfig(
		level=logging.DEBUG if verbose else logging.INFO,
		format="%(levelname)s: %(message)s",
	)


def _resolve_device(device_arg: str) -> str:
	# Picks the device depending on what the machine supports.
	if device_arg.lower() != "auto":
		return device_arg
	if torch is not None and torch.cuda.is_available():
		return "cuda"
	return "cpu"


def _load_sam2_predictor(config_path: Path, checkpoint_path: Path, device_arg: str) -> Sam2Runtime | None:
	# Loads the optional SAM2 model when the files are present.
	try:
		hydra_utils = importlib.import_module("hydra.utils")
		omegaconf_mod = importlib.import_module("omegaconf")
		sam_module = importlib.import_module("sam2.sam2_image_predictor")
	except ImportError:
		LOGGER.error(
			"SAM2 dependencies are missing. Install 'sam2', 'hydra-core', 'omegaconf', and 'torch' as listed in setup_venv/requirements.txt."
		)
		return None
	instantiate = getattr(hydra_utils, "instantiate")
	OmegaConf = getattr(omegaconf_mod, "OmegaConf")
	SAM2ImagePredictor = getattr(sam_module, "SAM2ImagePredictor")

	if torch is None:
		LOGGER.error("PyTorch is not installed; SAM2 cannot run without torch.")
		return None

	config_path = config_path.expanduser()
	checkpoint_path = checkpoint_path.expanduser()
	if not config_path.is_file():
		LOGGER.error("SAM2 config not found: %s", config_path)
		return None
	if not checkpoint_path.is_file():
		LOGGER.error("SAM2 checkpoint not found: %s", checkpoint_path)
		return None

	device = _resolve_device(device_arg)
	LOGGER.info("Loading SAM2 model (config=%s, checkpoint=%s, device=%s)", config_path, checkpoint_path, device)
	try:
		cfg = OmegaConf.load(str(config_path))
		OmegaConf.resolve(cfg)
		if "model" not in cfg:
			raise ValueError("Configuration file is missing the 'model' key required for instantiation.")
		model = instantiate(cfg["model"], _recursive_=True)
		state_dict = _load_checkpoint_state(checkpoint_path)
		missing_keys, unexpected_keys = model.load_state_dict(state_dict)
		if missing_keys:
			raise RuntimeError(f"Checkpoint is missing keys: {missing_keys}")
		if unexpected_keys:
			raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected_keys}")
		model = model.to(device)
		model.eval()
		predictor = SAM2ImagePredictor(model)
		return Sam2Runtime(predictor=predictor, device=device)
	except Exception as exc:  # noqa: BLE001 - bubble up helpful message
		LOGGER.exception("Failed to initialize SAM2: %s", exc)
		return None


def _load_checkpoint_state(checkpoint_path: Path) -> dict:
	# Pulls the model weights from the checkpoint file.
	weights = torch.load(str(checkpoint_path), map_location="cpu")
	if isinstance(weights, dict):
		model_state = weights.get("model")
		if isinstance(model_state, dict):
			return model_state
		return weights
	raise TypeError("Checkpoint did not contain a valid state dictionary.")


def _segment_with_aruco(
	image: np.ndarray,
	iterations: int,
	expansion: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
	# Runs the ArUco baseline to build a reference mask.
	detection = aruco_module._detect_aruco_markers(image)
	if detection is None:
		LOGGER.warning("Skipping image because ArUco markers were not detected.")
		return None

	mask = aruco_module._build_grabcut_mask(image.shape[:2], detection.boundary, expansion)
	segmentation = aruco_module._run_grabcut(image, mask, iterations)
	contour = aruco_module._extract_primary_contour(segmentation)
	if contour is None:
		LOGGER.warning("Could not extract a contour from the ArUco segmentation.")
		return None
	return detection.boundary, segmentation, contour


def _compute_bounding_box(
	hull: np.ndarray,
	image_shape: Tuple[int, int],
	scale: float,
) -> np.ndarray:
	# Expands the hull into a padded box for SAM2.
	x_min = float(np.min(hull[:, 0]))
	y_min = float(np.min(hull[:, 1]))
	x_max = float(np.max(hull[:, 0]))
	y_max = float(np.max(hull[:, 1]))
	width = x_max - x_min
	height = y_max - y_min
	pad_x = width * max(scale, 0.0)
	pad_y = height * max(scale, 0.0)
	w, h = image_shape
	bbox = np.array(
		[
			max(0.0, x_min - pad_x),
			max(0.0, y_min - pad_y),
			min(w - 1.0, x_max + pad_x),
			min(h - 1.0, y_max + pad_y),
		],
		dtype=np.float32,
	)
	return bbox


def _build_prompt_points(hull: np.ndarray, count: int, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
	# Samples a few positive clicks along the hull.
	if count <= 0 or hull.size == 0:
		return None, None
	count = min(count, len(hull))
	indices = np.linspace(0, len(hull) - 1, num=count, dtype=int)
	selected = hull[indices]
	centroid = hull.mean(axis=0, keepdims=True)
	points = np.vstack([selected, centroid])
	w, h = image_shape
	points[:, 0] = np.clip(points[:, 0], 0, w - 1)
	points[:, 1] = np.clip(points[:, 1], 0, h - 1)
	labels = np.ones(len(points), dtype=np.int32)
	return points.astype(np.float32), labels.astype(np.int32)



def _segment_with_sam2(
	runtime: Sam2Runtime,
	image: np.ndarray,
	bbox: np.ndarray,
	hull: np.ndarray,
	aruco_mask: np.ndarray,
	prompt_points: int,
) -> Tuple[np.ndarray, float] | Tuple[None, None]:
	# Calls SAM2 to get masks and keeps the one closest to ArUco.
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	runtime.predictor.set_image(image_rgb)
	point_coords, point_labels = _build_prompt_points(hull, prompt_points, image.shape[1::-1])
	predict_kwargs: dict[str, Any] = {"box": bbox, "multimask_output": True}
	if point_coords is not None and point_labels is not None:
		predict_kwargs["point_coords"] = point_coords
		predict_kwargs["point_labels"] = point_labels
	try:
		masks, scores, _logits = runtime.predictor.predict(**predict_kwargs)
	except Exception as exc:
		LOGGER.exception("SAM2 prediction failed: %s", exc)
		return None, None
	if masks is None or len(masks) == 0:
		return None, None
	masks_u8 = [(mask * 255).astype(np.uint8) for mask in masks]
	ious = [
		_calculate_iou(aruco_mask, cand)
		for cand in masks_u8
	]
	best_idx = int(np.argmax(ious))
	best_score = float(scores[best_idx]) if scores is not None else 0.0
	return masks_u8[best_idx], best_score


def _calculate_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
	# Compares two binary masks fairly.
	a = mask_a > 0
	b = mask_b > 0
	union = np.logical_or(a, b).sum()
	if union == 0:
		return 0.0
	intersection = np.logical_and(a, b).sum()
	return float(intersection / union)


def _draw_mask_panel(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
	# Builds a colored overlay so the mask pops out.
	panel = image.copy()
	overlay = np.zeros_like(panel)
	overlay[mask > 0] = color
	panel = cv2.addWeighted(panel, 0.7, overlay, 0.3, 0)
	contour = aruco_module._extract_primary_contour(mask)
	if contour is not None:
		cv2.drawContours(panel, [contour], -1, color, 2)
	return panel


def _compose_visualization(
	image: np.ndarray,
	aruco_contour: np.ndarray,
	sam_mask: np.ndarray,
	aruco_mask: np.ndarray,
	iou: float,
) -> np.ndarray:
	# Stacks the reference, SAM2, and blended views side by side.
	aruco_panel = aruco_module._draw_boundary(image, aruco_contour)
	sam_panel = _draw_mask_panel(image, sam_mask, (255, 0, 255))
	combined = image.copy()
	if aruco_contour is not None:
		cv2.drawContours(combined, [aruco_contour], -1, (0, 255, 0), 2)
	sam_contour = aruco_module._extract_primary_contour(sam_mask)
	if sam_contour is not None:
		cv2.drawContours(combined, [sam_contour], -1, (255, 0, 255), 2)
	panels = [
		("ArUco GrabCut", aruco_panel),
		("SAM2", sam_panel),
		((f"IoU: {iou:.3f}"), combined),
	]
	for text, panel in panels:
		cv2.putText(panel, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
	return np.concatenate([panel for _, panel in panels], axis=1)


def _iter_image_files(folder: Path, suffix: str) -> Iterable[Path]:
	# Finds supported images that have not been processed yet.
	for path in sorted(folder.iterdir()):
		if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not path.stem.endswith(suffix):
			yield path


def _process_single_image(
	image_path: Path,
	runtime: Sam2Runtime,
	suffix: str,
	aruco_iterations: int,
	aruco_expansion: float,
	bbox_scale: float,
	prompt_points: int,
) -> bool:
	# Handles one photo from read to save.
	LOGGER.info("Processing %s", image_path.name)
	image = cv2.imread(str(image_path))
	if image is None:
		LOGGER.error("Failed to read %s", image_path)
		return False

	aruco_result = _segment_with_aruco(image, aruco_iterations, aruco_expansion)
	if aruco_result is None:
		return False
	hull, aruco_mask, aruco_contour = aruco_result
	bbox = _compute_bounding_box(hull, image.shape[1::-1], bbox_scale)
	sam_mask, sam_score = _segment_with_sam2(runtime, image, bbox, hull, aruco_mask, prompt_points)
	if sam_mask is None:
		LOGGER.warning("SAM2 failed to segment %s", image_path.name)
		return False

	iou = _calculate_iou(aruco_mask, sam_mask)
	visualization = _compose_visualization(image, aruco_contour, sam_mask, aruco_mask, iou)
	output_path = image_path.with_name(f"{image_path.stem}{suffix}{image_path.suffix}")
	if cv2.imwrite(str(output_path), visualization):
		LOGGER.info("Saved %s (IoU=%.3f, SAM-score=%.3f)", output_path.name, iou, sam_score or 0.0)
		return True
	LOGGER.error("Failed to persist result for %s", image_path.name)
	return False


def _run(
	image_folder: Path,
	suffix: str,
	aruco_iterations: int,
	aruco_expansion: float,
	bbox_scale: float,
	prompt_points: int,
	runtime: Sam2Runtime,
) -> int:
	# Walks the folder and tracks how many runs succeed.
	if not image_folder.is_dir():
		LOGGER.error("%s is not a valid folder", image_folder)
		return 1
	image_paths = list(_iter_image_files(image_folder, suffix))
	if not image_paths:
		LOGGER.error("No supported image files were found in %s", image_folder)
		return 1

	successes = 0
	for image_path in image_paths:
		try:
			if _process_single_image(
				image_path,
				runtime,
				suffix,
				aruco_iterations,
				aruco_expansion,
				bbox_scale,
				prompt_points,
			):
				successes += 1
		except cv2.error as exc:
			LOGGER.exception("OpenCV error while processing %s: %s", image_path.name, exc)
		except Exception as exc:  # noqa: BLE001
			LOGGER.exception("Unexpected error while processing %s: %s", image_path.name, exc)

	if successes == 0:
		LOGGER.warning("SAM2 pipeline failed for all files.")
		return 2
	LOGGER.info("Completed %s/%s files.", successes, len(image_paths))
	return 0


def main(argv: Sequence[str] | None = None) -> int:
	# Acts as the entry point for the script.
	parser = _build_argument_parser()
	args = parser.parse_args(argv)
	_setup_logging(args.verbose)
	runtime = _load_sam2_predictor(args.model_config, args.checkpoint, args.device)
	if runtime is None:
		return 1
	return _run(
		args.image_folder,
		args.suffix,
		args.aruco_iterations,
		args.aruco_hull_expansion,
		args.bbox_scale,
		args.prompt_points,
		runtime,
	)


if __name__ == "__main__":
	sys.exit(main())
