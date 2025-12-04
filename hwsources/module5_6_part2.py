"""
CSC 8830 Computer Vision
Avyuktkrishna Ramasamy
Dr. Ashwin Ashok
Module 5-6 Assignment Part 2 - Real time Object Tracker (without Aruco markers)

The purpose of this script is to track a moving object 
in real time WITHOUT the use of Aruco markers. The script 
relies on using the segmentation features of SAM2. By using 
SAM2's temporal awarness features the script identifies a target
object in the initial frame and propagates the segmentation mask 
throughout the video in real time. Even partially covered objects 
or rotating objects which traditional geometric markers may be 
unable to maintain tracking, can be tracked with this script

The outputs of the program include
 - originalfilename-tracked.mp4 :- final video with segmentation overlay (bounding box)
 - originalfilename-masks.npz :- compressed numpy archive with the mask sequence

Usage:
    python module5_6_part2.py ./data/video.mp4 --device cuda --resize 0.5
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

# Opens the requested video file for reading frames later
def open_video_capture(path: str):
    return cv2.VideoCapture(path)


# Makes sure the given file path exists before continuing
def validate_input_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    candidate = Path(path)
    return str(candidate) if candidate.is_file() else None


# Creates a video writer with reasonable codec fallbacks based on file suffix
def ensure_writer(output_path: Optional[str], fps: float, w: int, h: int) -> Optional[cv2.VideoWriter]:
    if not output_path:
        return None

    resolved = str(output_path)
    suffix = Path(resolved).suffix.lower()
    codec_candidates = {
        ".mp4": ["avc1", "H264", "mp4v"],
        ".mov": ["avc1", "H264", "mp4v"],
        ".m4v": ["avc1", "H264", "mp4v"],
        ".avi": ["XVID", "mp4v"],
    }.get(suffix, ["mp4v"])

    for code in codec_candidates:
        writer = cv2.VideoWriter(resolved, cv2.VideoWriter_fourcc(*code), fps, (w, h))
        if writer.isOpened():
            print(f"[module5_6_part2] Using codec {code} for output {resolved}")
            return writer
        writer.release()
        print(f"[module5_6_part2] Failed to open VideoWriter with codec {code}, trying next candidate…")

    print(f"[module5_6_part2] Could not initialize a VideoWriter for {resolved}.")
    return None


# Paints a translucent mask and bounding box onto the frame
def overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.25):
    mask_bool = mask.astype(bool, copy=False)
    overlay = frame.copy()
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + np.array(color, dtype=np.uint8) * alpha).astype(np.uint8)
    frame[:] = overlay

    ys, xs = np.where(mask_bool)
    if ys.size and xs.size:
        cv2.rectangle(frame, (xs.min(), ys.min()), (xs.max(), ys.max()), color, 2)


# Resizes a mask so it lines up with the current frame
def _match_mask_to_frame(mask: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] != frame_shape:
        mask = cv2.resize(mask.astype(np.uint8), (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool, copy=False)


# Replays the original video while overlaying the stored masks
def render_masks(cap: cv2.VideoCapture, writer: Optional[cv2.VideoWriter], masks: np.ndarray):
    if masks.ndim == 3 and masks.shape[-1] < min(masks.shape[0], masks.shape[1]):
        masks = np.moveaxis(masks, -1, 0)

    total_frames = masks.shape[0]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask = masks[idx] if idx < total_frames else masks[-1]
        overlay_mask(frame, _match_mask_to_frame(mask, frame.shape[:2]))
        if writer:
            writer.write(frame)
        idx += 1

    cap.release()
    if writer:
        writer.release()


# Finds a usable SAM checkpoint from args, env, or common cache paths
def _resolve_checkpoint(checkpoint: Optional[str]) -> str:
    checkpoint_path = checkpoint or os.environ.get("SAM_CHECKPOINT")
    if checkpoint_path is None:
        for candidate in (
            Path.home() / ".cache" / "sam" / "sam_vit_b.pth",
            Path("hwsources/resources/m5_6/sam_vit_b.pth"),
        ):
            if Path(candidate).is_file():
                checkpoint_path = str(candidate)
                break
    if not checkpoint_path:
        raise RuntimeError("No SAM checkpoint found. Provide --checkpoint or set SAM_CHECKPOINT.")
    return checkpoint_path


# Chooses the SAM model variant that matches the checkpoint name
def _select_model_key(checkpoint: str, registry_keys: Sequence[str]):
    stem = Path(checkpoint).stem.lower()
    for tag in ("vit_h", "vit_l", "vit_b"):
        if tag in stem:
            match = next((k for k in registry_keys if tag in k.lower()), None)
            if match:
                return match
    for prefer in ("vit_h", "vit_l", "vit_b"):
        match = next((k for k in registry_keys if prefer in k.lower()), None)
        if match:
            return match
    return next(iter(registry_keys))


# Loads SAM and moves it to the requested device when possible
def _load_sam_model(checkpoint: str, device: str):
    try:
        from segment_anything import sam_model_registry
    except Exception as exc:
        raise RuntimeError("segment_anything is not installed or could not be imported") from exc

    model_key = _select_model_key(checkpoint, sam_model_registry.keys())
    sam_model = sam_model_registry[model_key](checkpoint=checkpoint)
    try:
        sam_model.to({"mps": "mps", "cuda": "cuda"}.get(device, "cpu"))
    except Exception:
        pass
    return sam_model


# Converts a binary mask into a tight [x0, y0, x1, y1] box
def _mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if not ys.size or not xs.size:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


# Scores automatic masks based on size and SAM confidence
def _choose_segment(segments, confidences, frame_area: int, target_ratio: float, min_ratio: float, max_ratio: float):
    best = None
    best_score = float("-inf")
    for seg, conf in zip(segments, confidences):
        if not frame_area:
            continue
        ratio = float(seg.sum()) / frame_area
        penalty = 0.0
        if ratio < min_ratio:
            penalty = min_ratio - ratio
        elif ratio > max_ratio:
            penalty = ratio - max_ratio
        score = conf - abs(ratio - target_ratio) - penalty
        if score > best_score:
            best_score = score
            best = (seg, ratio, conf)
    return best


# Measures how large a mask is relative to the frame
def _mask_ratio(mask: np.ndarray) -> float:
    return float(mask.sum()) / mask.size if mask.size else 0.0


# Computes IoU between two masks to keep tracking stable
def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (inter / union) if union else 0.0


# Predicts the next mask by focusing SAM around the previous box
def _predict_from_last(predictor, last_mask: np.ndarray):
    box = _mask_to_box(last_mask)
    if box is None:
        return None
    try:
        masks_pred, scores_pred, _ = predictor.predict(box=box[None, :], point_coords=None, point_labels=None, multimask_output=False)
        mask = masks_pred[0].astype(bool)
        return mask, _mask_ratio(mask), float(scores_pred[0])
    except Exception:
        return None


    # Falls back to SAM's automatic mask proposals when tracking is lost
def _reacquire_mask(mask_generator, frame_rgb: np.ndarray, frame_area: int, last_ratio: Optional[float],
                    min_ratio: float, max_ratio: float, default_ratio: float):
    try:
        masks_data = mask_generator.generate(frame_rgb)
    except Exception:
        return None
    if not masks_data:
        return None
    segments = [entry["segmentation"].astype(bool) for entry in masks_data]
    confidences = [float(entry.get("predicted_iou", entry.get("stability_score", 1.0))) for entry in masks_data]
    target_ratio = last_ratio if last_ratio is not None else default_ratio
    choice = _choose_segment(segments, confidences, frame_area, target_ratio, min_ratio, max_ratio)
    return choice


# Generates a boolean mask for each frame using SAM plus lightweight heuristics
def generate_masks_with_sam(video_path: str, device: str = "cpu", resize_scale: float = 1.0, checkpoint: Optional[str] = None):
    try:
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor
    except Exception as exc:
        raise RuntimeError("segment_anything is not installed or could not be imported") from exc

    checkpoint_path = _resolve_checkpoint(checkpoint)
    sam_model = _load_sam_model(checkpoint_path, device)
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    predictor = SamPredictor(sam_model)

    cap = open_video_capture(video_path)
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Failed to open input {video_path}")

    MIN_AREA_RATIO = 0.002
    MAX_AREA_RATIO = 0.45
    MIN_CONFIDENCE = 0.35
    MIN_IOU = 0.2
    MAX_TRACK_GAP = 5
    AREA_SIMILARITY = 0.12
    INITIAL_TARGET_RATIO = 0.08

    masks = []
    processed = 0
    last_mask_small: Optional[np.ndarray] = None
    last_area_ratio: Optional[float] = None
    miss_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if resize_scale != 1.0:
            small_frame = cv2.resize(frame, (int(frame.shape[1] * resize_scale), int(frame.shape[0] * resize_scale)))
        else:
            small_frame = frame
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)
        frame_area = small_frame.shape[0] * small_frame.shape[1]

        candidate = _predict_from_last(predictor, last_mask_small) if last_mask_small is not None else None
        if candidate is None or miss_counter >= MAX_TRACK_GAP:
            candidate = _reacquire_mask(mask_generator, frame_rgb, frame_area, last_area_ratio,
                                        MIN_AREA_RATIO, MAX_AREA_RATIO, INITIAL_TARGET_RATIO)
            if candidate is not None:
                miss_counter = 0

        mask, ratio, score = (candidate if candidate is not None else (None, 0.0, 0.0))
        valid = mask is not None and MIN_AREA_RATIO <= ratio <= MAX_AREA_RATIO and score >= MIN_CONFIDENCE
        if valid and last_area_ratio is not None:
            valid = abs(ratio - last_area_ratio) <= AREA_SIMILARITY
        if valid and mask is not None and last_mask_small is not None:
            valid = _mask_iou(mask, last_mask_small) >= MIN_IOU

        if valid and mask is not None:
            stable_small = mask
            miss_counter = 0
        elif last_mask_small is not None and miss_counter < MAX_TRACK_GAP:
            stable_small = last_mask_small
            miss_counter += 1
        else:
            stable_small = np.zeros(small_frame.shape[:2], dtype=bool)
            miss_counter += 1
            if miss_counter >= MAX_TRACK_GAP:
                last_mask_small = None
                last_area_ratio = None

        if stable_small.any():
            last_mask_small = stable_small.copy()
            last_area_ratio = _mask_ratio(last_mask_small)

        masks.append(_match_mask_to_frame(stable_small, frame.shape[:2]).astype(np.uint8))
        processed += 1
        if processed % 50 == 0:
            print(f"SAM tracker processed {processed} frames…")

    cap.release()
    if not masks:
        raise RuntimeError("No frames were processed from the input video.")
    return np.stack(masks, axis=0)


# Handles CLI parsing, mask creation, saving, and visualization
def main(argv=None):
    parser = argparse.ArgumentParser(description="Markerless SAM tracker (video-only)")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu", help="Device for SAM model (if available)")
    parser.add_argument("--resize", type=float, default=1.0, help="Scale frames for segmentation (0.0-1.0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SAM checkpoint (overrides env var)")
    args = parser.parse_args(argv)

    if not (0.0 < args.resize <= 1.0):
        print("--resize must be within (0.0, 1.0].")
        sys.exit(1)

    input_path = validate_input_path(args.input)
    if not input_path:
        print("Invalid or missing input path. Please pass a valid video file path.")
        sys.exit(1)

    cap = open_video_capture(input_path)
    if not cap or not cap.isOpened():
        print(f"Failed to open input {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    in_path = Path(input_path)
    masks_path = in_path.with_name(f"{in_path.stem}-masks.npz")
    output_path = in_path.with_name(f"{in_path.stem}-tracked.mp4")

    writer = ensure_writer(str(output_path), fps, w, h)
    if writer is None:
        raise RuntimeError(f"Unable to initialize video writer for {output_path}. Please install H.264/MP4 codecs.")

    try:
        print("Generating masks with SAM tracker…")
        masks = generate_masks_with_sam(input_path, device=args.device, resize_scale=args.resize, checkpoint=args.checkpoint)
    except Exception as exc:
        raise RuntimeError(f"SAM tracking failed: {exc}") from exc

    print(f"Saving masks NPZ to {masks_path}")
    np.savez_compressed(str(masks_path), masks=masks.astype(np.uint8))

    cap2 = open_video_capture(input_path)
    render_masks(cap2, writer, masks)


if __name__ == "__main__":
    main()
