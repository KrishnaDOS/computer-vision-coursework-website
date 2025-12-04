import base64
import csv
import io
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, url_for, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import HW source modules
from hwsources.module1 import calculate_focal_length, calculate_real_dimension
import hwsources.module2_part1 as module2_part1
import hwsources.module5_6_part1 as module5_6
from hwsources.module2_part2 import process_image as module2_process_image
import hwsources.module2_part3 as module2_part3
import hwsources.module3_part1 as module3_part1
import hwsources.module3_part2 as module3_part2
import hwsources.module4_part1 as module4_part1
import hwsources.module4_part2 as module4_part2
import hwsources.module7_part2 as module7_part2
from hwsources.module7_part1 import (
    compute_focal_length_from_reference,
    compute_depth_from_stereo,
    compute_real_size_from_depth,
)

# Serve static assets from the templates folder so they can be moved
# from /static into /templates while keeping the same "url_for('static', ...)"
app = Flask(__name__, static_folder='templates', static_url_path='/static')
CORS(app)  # Enable CORS for all routes

ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
PROJECT_ROOT = Path(__file__).resolve().parent
MODULE2_PART1_DIR = PROJECT_ROOT / 'hwsources' / 'resources' / 'm2'
MODULE2_PART1_SCENE = MODULE2_PART1_DIR / 'scene.jpg'
MODULE2_PART3_TEMPLATE_LIMIT = 25
INSTRUCTIONS_DIR = PROJECT_ROOT / 'hwinstructions'
MODULE2_PART2_SAMPLE = MODULE2_PART1_DIR / 'tree.jpg'
MODULE1_DIR = PROJECT_ROOT / 'hwsources' / 'resources' / 'm1'
MODULE3_SAMPLE_DIR = PROJECT_ROOT / 'hwsources' / 'resources' / 'm3'
MODULE3_MIN_IMAGES = 10
MODULE3_PREVIEW_MAX_DIM = 1400
MODULE3_PART3_PROCESS_MAX_DIM = 2000
MODULE3_PART4_SAMPLE_DIR = MODULE3_SAMPLE_DIR / 'part2'
MODULE3_PART5_OUTPUT_DIR = MODULE3_SAMPLE_DIR / 'part3-sam-outputs'
MODULE56_DIR = PROJECT_ROOT / 'hwsources' / 'resources' / 'm5_6'
MODULE56_SAMPLE = MODULE56_DIR / 'aruco-marker.mp4'
MODULE56_PART2_VIDEO = MODULE56_DIR / 'iphone-moving.mp4'
MODULE56_PART2_MASKS = MODULE56_DIR / 'iphone-moving-masks.npz'
MODULE4_MIN_IMAGES = 8
MODULE4_SAMPLE_DIR = PROJECT_ROOT / 'hwsources' / 'resources' / 'm4'
MODULE7_RESOURCES_DIR = PROJECT_ROOT / 'hwsources' / 'resources' / 'm7'
MODULE7_PART2_SAMPLE = MODULE7_RESOURCES_DIR / 'karate.mp4'
MODULE7_PART1_DIR = MODULE7_RESOURCES_DIR / 'part1'
MODULE7_PART1_REFERENCE = MODULE7_PART1_DIR / 'ref.jpeg'
MODULE7_PART1_LEFT = MODULE7_PART1_DIR / 'left.jpeg'
MODULE7_PART1_RIGHT = MODULE7_PART1_DIR / 'right.jpeg'
MODULE7_PART1_MEASUREMENTS = MODULE7_PART1_DIR / 'measurements.md'
MODULE2_PREVIEW_MAX_DIM = 1400
MODULE2_JPEG_QUALITY = 85

VIDEO_MIME_MAP = {
    '.mp4': 'video/mp4',
    '.webm': 'video/webm',
    '.avi': 'video/x-msvideo',
    '.mov': 'video/quicktime',
    '.mkv': 'video/x-matroska',
}

MODULE_INSTRUCTION_FILES = {
    'a1': 'Assignment1.pdf',
    'module1': 'Assignment1.pdf',
    '1': 'Assignment1.pdf',
    'a2': 'Assignment2.pdf',
    'module2': 'Assignment2.pdf',
    '2': 'Assignment2.pdf',
    'a3': 'Assignment3.pdf',
    'module3': 'Assignment3.pdf',
    '3': 'Assignment3.pdf',
    'a4': 'Assignment4.pdf',
    'module4': 'Assignment4.pdf',
    '4': 'Assignment4.pdf',
    'a56': 'Assignment5-6.pdf',
    'module5': 'Assignment5-6.pdf',
    'module6': 'Assignment5-6.pdf',
    '5': 'Assignment5-6.pdf',
    '6': 'Assignment5-6.pdf',
    'a7': 'Assignment7.pdf',
    'module7': 'Assignment7.pdf',
    '7': 'Assignment7.pdf',
}

WRITTEN_ANSWERS_DIR = PROJECT_ROOT / 'hwwrittenanswers'
WRITTEN_ANSWER_FILES = {
    'a7': {
        'q2': 'M7_Written_Question.pdf',
        'question2': 'M7_Written_Question.pdf',
        '2': 'M7_Written_Question.pdf',
    },
    'a56': {
        'q1': 'M56_Written_Question.pdf',
    },
}

WRITTEN_MODULE_ALIASES = {
    'module7': 'a7',
    'assignment7': 'a7',
    '7': 'a7',
    'module5': 'a56',
    'module6': 'a56',
    'module5&6': 'a56',
    'module5-6': 'a56',
    'assignment5': 'a56',
    'assignment6': 'a56',
    'assignment5&6': 'a56',
    'assignment5-6': 'a56',
    'a56': 'a56',
    '5': 'a56',
    '6': 'a56',
}

WRITTEN_QUESTION_ALIASES = {
    'question-2': 'q2',
    'question_2': 'q2',
    'question2': 'q2',
    '2': 'q2',
    'question-1': 'q1',
    'question_1': 'q1',
    'question1': 'q1',
    '1': 'q1',
    'q1': 'q1',
}


def _image_file_to_data_url(path: Path) -> str:
    """Return a base64 data URL for the given image path."""
    suffix = path.suffix.lower()
    mime = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.tif': 'image/tiff',
        '.tiff': 'image/tiff'
    }.get(suffix, 'application/octet-stream')

    data = path.read_bytes()
    encoded = base64.b64encode(data).decode('ascii')
    return f"data:{mime};base64,{encoded}"


def _video_file_to_data_url(path: Path) -> str:
    """Return a base64 video data URL for the given file."""
    mime = VIDEO_MIME_MAP.get(path.suffix.lower(), 'application/octet-stream')
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode('ascii')
    return f"data:{mime};base64,{encoded}"


def _csv_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode('ascii')
    return f"data:text/csv;base64,{encoded}"


def _build_csv_preview(csv_path: Path, limit: int = 5):
    columns = []
    rows = []
    total = 0
    with csv_path.open('r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = reader.fieldnames or []
        for row in reader:
            total += 1
            if len(rows) < limit:
                rows.append({col: row.get(col) for col in columns})
    return columns, rows, total


def _get_request_payload():
    return request.form if request.form else (request.get_json(silent=True) or {})


def _truthy_flag(value) -> bool:
    """Return True for common truthy strings/values."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    return value_str in {'1', 'true', 'yes', 'sample', 'example', 'on'}


def _get_module3_files_from_request():
    files = request.files.getlist('images')
    if not files:
        files = request.files.getlist('images[]')
    return files


def _list_module3_sample_paths():
    if not MODULE3_SAMPLE_DIR.exists():
        raise FileNotFoundError('Module 3 example dataset is missing.')

    sample_paths = [
        path for path in MODULE3_SAMPLE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]

    if len(sample_paths) < MODULE3_MIN_IMAGES:
        raise ValueError(f"Example dataset must contain at least {MODULE3_MIN_IMAGES} images.")

    sample_paths.sort(key=lambda path: path.name.lower())
    return sample_paths


def _load_module3_sample_images():
    uploads = []
    for path in _list_module3_sample_paths():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Failed to read example image {path.name}.')
        uploads.append({'filename': path.name, 'image': image})
    return uploads


def _list_module3_part4_sample_paths():
    if not MODULE3_PART4_SAMPLE_DIR.exists():
        raise FileNotFoundError('Module 3 Part 4 example dataset is missing.')

    sample_paths = [
        path for path in MODULE3_PART4_SAMPLE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            and '_detect' not in path.name.lower()
        )
    ]

    if len(sample_paths) < MODULE3_MIN_IMAGES:
        raise ValueError(f"Module 3 Part 4 example dataset must contain at least {MODULE3_MIN_IMAGES} images.")

    sample_paths.sort(key=lambda path: path.name.lower())
    return sample_paths


def _load_module3_part4_sample_images():
    uploads = []
    for path in _list_module3_part4_sample_paths():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Failed to read Module 3 Part 4 example image {path.name}.')
        uploads.append({'filename': path.name, 'image': image})
    return uploads


def _list_module3_part5_pairs():
    if not MODULE3_PART4_SAMPLE_DIR.exists():
        raise FileNotFoundError('Module 3 Part 5 source images are missing (part2 folder).')
    if not MODULE3_PART5_OUTPUT_DIR.exists():
        raise FileNotFoundError('Module 3 Part 5 SAM outputs are missing (part3-sam-outputs folder).')

    originals = [
        path for path in MODULE3_PART4_SAMPLE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            and '_detect' not in path.name.lower()
        )
    ]

    if not originals:
        raise ValueError('No source images were found for the Module 3 Part 5 showcase.')

    originals.sort(key=lambda path: _sort_key_natural(path.name))
    pairs = []
    for original_path in originals:
        target_name = f"{original_path.stem}_sam2{original_path.suffix}"
        sam_path = MODULE3_PART5_OUTPUT_DIR / target_name
        if not sam_path.exists():
            continue
        pairs.append((original_path, sam_path))

    if not pairs:
        raise ValueError('SAM2 showcase outputs are missing. Populate part3-sam-outputs with *_sam2 files.')

    return pairs


def _load_module3_uploads(file_storages):
    uploads = []
    for idx, storage in enumerate(file_storages, start=1):
        if not storage or not storage.filename:
            continue

        filename = secure_filename(storage.filename) or f'image_{idx:02d}.png'
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(f'Unsupported image format for {filename}.')

        data = storage.read()
        if not data:
            raise ValueError(f'Uploaded file {filename} is empty.')

        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Could not decode {filename}. Please upload a valid image file.')

        uploads.append({'filename': filename, 'image': image})

    if len(uploads) < MODULE3_MIN_IMAGES:
        raise ValueError(f"Please upload at least {MODULE3_MIN_IMAGES} images or click 'Use example data'.")

    return uploads


def _collect_module3_inputs():
    payload = _get_request_payload()
    use_sample = False

    if payload:
        for key in ('sample', 'useSample', 'use_sample', 'dataset'):
            if key in payload and _truthy_flag(payload.get(key)):
                use_sample = True
                break

    if use_sample:
        return _load_module3_sample_images(), 'sample'

    files = _get_module3_files_from_request()
    if not files:
        raise ValueError(f"Please upload at least {MODULE3_MIN_IMAGES} images or click 'Use example data'.")

    return _load_module3_uploads(files), 'upload'


def _collect_module3_part4_inputs():
    payload = _get_request_payload()
    use_sample = False

    if payload:
        for key in ('sample', 'useSample', 'samplePart4', 'part4Sample'):
            if key in payload and _truthy_flag(payload.get(key)):
                use_sample = True
                break

    if use_sample:
        return _load_module3_part4_sample_images(), 'sample'

    files = _get_module3_files_from_request()
    if not files:
        raise ValueError(f"Please upload at least {MODULE3_MIN_IMAGES} images or click 'Use example data'.")

    return _load_module3_uploads(files), 'upload'


def _run_module3_processor(uploads, processor, part_label, *, max_process_dim: int | None = None):
    results = []
    for entry in uploads:
        filename = entry['filename']
        source_image = entry['image']
        process_image = _resize_for_processing(source_image, max_process_dim)
        try:
            processed = processor(process_image)
        except Exception as exc:
            raise RuntimeError(f"Module 3 {part_label}: failed to process {filename}: {exc}") from exc

        if processed is None or getattr(processed, 'size', 0) == 0:
            raise RuntimeError(f"Module 3 {part_label}: processing returned an empty result for {filename}.")

        input_preview = _resize_for_preview(process_image, MODULE3_PREVIEW_MAX_DIM)
        output_preview = _resize_for_preview(processed, MODULE3_PREVIEW_MAX_DIM)

        results.append({
            'filename': filename,
            'inputImage': _image_array_to_data_url(input_preview, ext='.jpg'),
            'outputImage': _image_array_to_data_url(output_preview, ext='.jpg'),
        })

    return results


def _run_module3_part4_processor(uploads, iterations: int = 5, expansion: float = 0.15):
    results = []
    for entry in uploads:
        filename = entry['filename']
        image = entry['image']

        detection = module3_part2._detect_aruco_markers(image)
        if detection is None:
            raise RuntimeError(f"Module 3 Part 4: could not find at least three ArUco markers in {filename}.")

        mask = module3_part2._build_grabcut_mask(image.shape[:2], detection.boundary, expansion)
        segmentation = module3_part2._run_grabcut(image, mask, iterations)
        contour = module3_part2._extract_primary_contour(segmentation)
        if contour is None:
            raise RuntimeError(f"Module 3 Part 4: could not extract an object contour for {filename}.")

        overlay = module3_part2._draw_boundary(image, contour)

        marker_ids = detection.ids
        if isinstance(marker_ids, np.ndarray):
            marker_ids = marker_ids.astype(int).reshape(-1).tolist()
        elif marker_ids is None:
            marker_ids = []
        else:
            marker_ids = [int(marker_ids)]

        input_preview = _resize_for_preview(image, MODULE3_PREVIEW_MAX_DIM)
        output_preview = _resize_for_preview(overlay, MODULE3_PREVIEW_MAX_DIM)

        results.append({
            'filename': filename,
            'inputImage': _image_array_to_data_url(input_preview, ext='.jpg'),
            'outputImage': _image_array_to_data_url(output_preview, ext='.jpg'),
            'markerCount': len(marker_ids),
            'markerIds': marker_ids,
        })

    return results


def _prepare_video_for_web(path: Path) -> Path:
    """If ffmpeg is available, transcode the video to H.264 + faststart for browser playback."""
    ffmpeg_bin = shutil.which('ffmpeg')
    if not ffmpeg_bin:
        return path

    temp_output = path.with_name(path.stem + '_web.mp4')
    cmd = [
        ffmpeg_bin,
        '-y',
        '-i', str(path),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-an',
        str(temp_output)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Replace original file so downstream code continues to reference the same path/name
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        temp_output.replace(path)
    except Exception as exc:
        # Clean up temp output on failure and fall back to original file
        if temp_output.exists():
            try:
                temp_output.unlink()
            except OSError:
                pass
        print(f"[app] ffmpeg transcode failed ({exc}), using original output.")
    return path


def _clear_module2_part1_results() -> int:
    """Remove generated match images from the Module 2 Part 1 resources folder."""
    if not MODULE2_PART1_DIR.exists():
        return 0

    deleted = 0
    for path in MODULE2_PART1_DIR.glob('*_matched_template_*'):
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            continue
    return deleted


def _image_array_to_data_url(image, ext: str = '.png', jpeg_quality: int = 90) -> str:
    """Encode an OpenCV image (np.ndarray) into a base64 data URL."""
    encode_ext = ext if ext.startswith('.') else f'.{ext}'
    mime = 'image/png'
    encode_params = []
    if encode_ext.lower() in {'.jpg', '.jpeg'}:
        mime = 'image/jpeg'
        quality = int(np.clip(jpeg_quality, 10, 100))
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    success, buffer = cv2.imencode(encode_ext, image, encode_params)
    if not success:
        raise ValueError('Failed to encode image buffer for response.')

    encoded = base64.b64encode(buffer.tobytes()).decode('ascii')
    return f"data:{mime};base64,{encoded}"


def _resize_for_preview(image, max_dim: int):
    """Return a resized copy of `image` so its largest dimension <= max_dim."""
    if image is None or getattr(image, 'size', 0) == 0:
        return image

    height, width = image.shape[:2]
    largest = max(height, width)
    if max_dim <= 0 or largest <= max_dim:
        return image

    scale = max_dim / float(largest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    # INTER_AREA handles down-sampling well.
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _image_array_to_preview_data_url(image, max_dim: int, *, jpeg_quality: int = MODULE2_JPEG_QUALITY) -> str:
    """Resize `image` to fit within `max_dim` and encode as a JPEG data URL."""
    preview = _resize_for_preview(image, max_dim)
    return _image_array_to_data_url(preview, ext='.jpg', jpeg_quality=jpeg_quality)


def _image_path_to_preview_data_url(path: Path, max_dim: int, *, jpeg_quality: int = MODULE2_JPEG_QUALITY) -> str:
    """Load an image from disk, resize for preview, and encode as JPEG data URL."""
    if not path.exists():
        raise FileNotFoundError(f"Preview image missing: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read preview image: {path}")
    return _image_array_to_preview_data_url(image, max_dim, jpeg_quality=jpeg_quality)


def _resize_for_processing(image, max_dim: int | None):
    """Downscale large inputs before heavy pipelines run to keep latency reasonable."""
    if image is None or getattr(image, 'size', 0) == 0 or not max_dim or max_dim <= 0:
        return image

    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_dim:
        return image

    scale = max_dim / float(largest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _sort_key_natural(name: str):
    """Return a natural sort key using module4's helper when available."""
    if hasattr(module4_part1, 'correct_number_sorting'):
        return module4_part1.correct_number_sorting(name)
    return name.lower()


def _load_module4_uploads(file_storages):
    """Read uploaded portrait images, enforce constraints, and return sorted entries."""
    files = [fs for fs in (file_storages or []) if fs and fs.filename]
    if not files:
        raise ValueError(f'Please upload at least {MODULE4_MIN_IMAGES} portrait images (height greater than width).')

    uploads = []
    for storage in files:
        filename = secure_filename(storage.filename) or 'upload.jpg'
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported file type for '{filename}'.")

        data = storage.read()
        if not data:
            raise ValueError(f"File '{filename}' was empty.")
        image_array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode '{filename}'. Did you upload a valid image?")

        height, width = image.shape[:2]
        if height <= width:
            raise ValueError(f"'{filename}' must be portrait oriented (height greater than width).")

        uploads.append({'filename': filename, 'image': image})

    if len(uploads) < MODULE4_MIN_IMAGES:
        raise ValueError(f'Please upload at least {MODULE4_MIN_IMAGES} portrait images.')

    uploads.sort(key=lambda entry: _sort_key_natural(entry['filename']))
    return uploads


def _list_module4_sample_paths():
    if not MODULE4_SAMPLE_DIR.exists():
        raise FileNotFoundError(f"Example image folder missing: {MODULE4_SAMPLE_DIR}")

    sample_paths = [
        path for path in MODULE4_SAMPLE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]

    if not sample_paths:
        raise FileNotFoundError(f"No example images found inside {MODULE4_SAMPLE_DIR}")

    sample_paths.sort(key=lambda path: _sort_key_natural(path.name))
    return sample_paths


def _load_module4_sample_images():
    sample_paths = _list_module4_sample_paths()
    uploads = []
    for path in sample_paths:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to read sample image: {path.name}")
        height, width = image.shape[:2]
        if height <= width:
            raise ValueError(f"Sample image '{path.name}' must be portrait oriented (height > width).")
        uploads.append({'filename': path.name, 'image': image})

    if len(uploads) < MODULE4_MIN_IMAGES:
        raise ValueError(f"Example dataset must contain at least {MODULE4_MIN_IMAGES} portrait images. Found {len(uploads)}.")

    return uploads


def _collect_module4_inputs():
    files = _get_module4_files_from_request()
    payload = request.form or (request.get_json(silent=True) or {})
    sample_flag = ''
    if payload:
        sample_flag = str(payload.get('sample') or '').strip().lower()

    if sample_flag in {'1', 'true', 'yes', 'sample', 'm4', 'example'}:
        return _load_module4_sample_images(), 'sample'

    if files:
        return _load_module4_uploads(files), 'upload'

    raise ValueError(f"Please upload at least {MODULE4_MIN_IMAGES} portrait images or click 'Load example data'.")


def _stitch_module4_part1(images):
    """Run the OpenCV (Module 4 Part 1) stitching pipeline and return a panorama image."""
    if len(images) < 4:
        raise ValueError('At least 4 images are required to compute the OpenCV panorama.')

    work_width = getattr(module4_part1, 'WORK_WIDTH', 800)
    erosion_iterations = getattr(module4_part1, 'SEAM_EROSION_ITERATIONS', 1)
    resized = [module4_part1.resize_image(img, target_width=work_width) for img in images]
    homographies = [np.identity(3)]

    for idx in range(len(resized) - 1):
        kps_prev, feats_prev = module4_part1.extract_sift_keypoints_descriptors(resized[idx])
        kps_curr, feats_curr = module4_part1.extract_sift_keypoints_descriptors(resized[idx + 1])
        result = module4_part1.match_keypoints_affine(kps_curr, kps_prev, feats_curr, feats_prev)
        if result is None:
            raise RuntimeError(f'Unable to align image {idx + 2} with image {idx + 1}. Not enough matches detected.')
        _, homography_curr_to_prev, _ = result
        homographies.append(homographies[idx].dot(homography_curr_to_prev))

    middle_index = len(resized) // 2
    homography_middle_to_0 = homographies[middle_index]
    homography_0_to_middle = np.linalg.inv(homography_middle_to_0)

    new_homographies = []
    all_corners = []
    for img, homography in zip(resized, homographies):
        new_h = homography_0_to_middle.dot(homography)
        new_homographies.append(new_h)
        height, width = img.shape[:2]
        corners = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, new_h)
        all_corners.append(warped)

    if not all_corners:
        raise RuntimeError('Failed to determine panorama canvas size.')

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
    output_width, output_height = x_max - x_min, y_max - y_min
    if output_width <= 0 or output_height <= 0:
        raise RuntimeError('Computed panorama dimensions were invalid.')

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    panorama = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    erosion_kernel = np.ones((3, 3), np.uint8)
    draw_order = sorted(range(len(resized)), key=lambda i: abs(i - middle_index), reverse=True)

    for idx in draw_order:
        final_h = translation.dot(new_homographies[idx])
        warped = cv2.warpPerspective(resized[idx], final_h, (output_width, output_height), flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=erosion_iterations)
        panorama[mask > 0] = warped[mask > 0]

    panorama = module4_part1.remove_black_borders(panorama)
    if panorama is None or panorama.size == 0:
        raise RuntimeError('OpenCV stitching finished without a valid panorama output.')
    return panorama


def _stitch_module4_part2(images):
    """Run the custom-from-scratch Module 4 Part 2 pipeline."""
    if len(images) < 4:
        raise ValueError('At least 4 images are required to compute the custom panorama.')

    work_width = getattr(module4_part2, 'WORK_WIDTH', 400)
    resized = [module4_part2.resize_image(img, width=work_width) for img in images]
    panorama = module4_part2.run_stitching_custom(resized)
    if panorama is None or panorama.size == 0:
        raise RuntimeError('Custom stitching failed to return an output image.')
    return panorama


_module56_part2_cache = None


def _load_module56_part2_assets():
    """Load the precomputed SAM2 masks and derive bounding boxes for each frame."""
    global _module56_part2_cache
    if _module56_part2_cache is not None:
        return _module56_part2_cache

    if not MODULE56_PART2_VIDEO.exists():
        raise FileNotFoundError(f"Part 2 video missing at {MODULE56_PART2_VIDEO}")
    if not MODULE56_PART2_MASKS.exists():
        raise FileNotFoundError(f"Mask data missing at {MODULE56_PART2_MASKS}")

    try:
        with np.load(MODULE56_PART2_MASKS) as data:
            if 'masks' not in data:
                raise ValueError('Mask file does not contain a "masks" array')
            masks = data['masks']
    except Exception as exc:
        raise RuntimeError(f'Failed to load SAM2 mask data: {exc}') from exc

    if masks.ndim < 3:
        raise ValueError('Unexpected mask array shape; expected (frames, height, width).')

    frame_count_masks = int(masks.shape[0])
    mask_height = int(masks.shape[1])
    mask_width = int(masks.shape[2]) if masks.ndim >= 3 else int(masks.shape[1])
    frame_height = mask_height
    frame_width = mask_width

    boxes = []
    for idx in range(frame_count_masks):
        mask = masks[idx]
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_bool = mask.astype(bool)
        entry = {'frame': idx, 'box': None}
        if mask_bool.any():
            ys, xs = np.where(mask_bool)
            x1 = int(xs.min())
            x2 = int(xs.max()) + 1  # treat as exclusive bounds for drawing convenience
            y1 = int(ys.min())
            y2 = int(ys.max()) + 1
            entry['box'] = [x1, y1, x2, y2]
            entry['normalized'] = {
                'x1': x1 / mask_width if mask_width else 0.0,
                'y1': y1 / mask_height if mask_height else 0.0,
                'x2': x2 / mask_width if mask_width else 0.0,
                'y2': y2 / mask_height if mask_height else 0.0,
            }
            entry['area'] = int((x2 - x1) * (y2 - y1))
        boxes.append(entry)

    fps = 30.0
    frame_count_video = frame_count_masks
    cap = cv2.VideoCapture(str(MODULE56_PART2_VIDEO))
    if cap is not None and cap.isOpened():
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        if fps_read:
            fps = float(fps_read)
        width_read = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height_read = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width_read:
            frame_width = int(width_read)
        if height_read:
            frame_height = int(height_read)
        count_read = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if count_read:
            frame_count_video = int(count_read)
        cap.release()
    else:
        try:
            cap.release()
        except Exception:
            pass

    duration_seconds = frame_count_video / fps if fps else None

    if mask_width and mask_height and frame_width and frame_height:
        for entry in boxes:
            norm = entry.get('normalized') or None
            if not norm:
                entry['box'] = None
                entry['area'] = 0
                continue
            x1 = int(round(norm['x1'] * frame_width))
            y1 = int(round(norm['y1'] * frame_height))
            x2 = int(round(norm['x2'] * frame_width))
            y2 = int(round(norm['y2'] * frame_height))
            x1, x2 = max(0, x1), max(0, x2)
            y1, y2 = max(0, y1), max(0, y2)
            entry['box'] = [x1, y1, x2, y2]
            entry['area'] = int(max(0, x2 - x1) * max(0, y2 - y1))

    _module56_part2_cache = {
        'videoFilename': MODULE56_PART2_VIDEO.name,
        'frameWidth': frame_width,
        'frameHeight': frame_height,
        'fps': float(round(fps, 4)),
        'frameCount': frame_count_video,
        'maskFrameCount': frame_count_masks,
        'durationSeconds': float(round(duration_seconds, 4)) if duration_seconds else None,
        'maskFilename': MODULE56_PART2_MASKS.name,
        'boxes': boxes,
    }
    return _module56_part2_cache


def _run_module2_part1(threshold: float):
    """Execute template matching for every template_* file and collect results."""
    if not MODULE2_PART1_SCENE.exists():
        raise FileNotFoundError(f"Scene file missing: {MODULE2_PART1_SCENE}")

    template_files = sorted(MODULE2_PART1_DIR.glob('template_*.png'))
    if not template_files:
        raise FileNotFoundError(
            "No template files found in resources/m2 (expected files starting with 'template_')."
        )

    _clear_module2_part1_results()

    matches = []
    for tpl in template_files:
        tpl_name = tpl.name
        output_path = MODULE2_PART1_SCENE.with_name(
            f"{MODULE2_PART1_SCENE.stem}_matched_{tpl.stem}{MODULE2_PART1_SCENE.suffix}"
        )

        try:
            matched = module2_part1.match_template(
                scene_path=str(MODULE2_PART1_SCENE),
                template_path=str(tpl),
                threshold=threshold,
                save_result=True,
                draw_all=True,
            )
        except Exception as exc:
            matches.append({
                'template': tpl_name,
                'matched': False,
                'error': str(exc),
                'outputImage': None,
            })
            continue

        if matched and output_path.exists():
            try:
                image_data = _image_path_to_preview_data_url(
                    output_path,
                    MODULE2_PREVIEW_MAX_DIM,
                    jpeg_quality=MODULE2_JPEG_QUALITY,
                )
            except FileNotFoundError:
                image_data = None
        else:
            image_data = None

        matches.append({
            'template': tpl_name,
            'matched': bool(matched and image_data),
            'error': None,
            'outputImage': image_data,
            'outputFilename': output_path.name if image_data else None,
        })

    try:
        scene_data = _image_path_to_preview_data_url(
            MODULE2_PART1_SCENE,
            MODULE2_PREVIEW_MAX_DIM,
            jpeg_quality=MODULE2_JPEG_QUALITY,
        )
    except FileNotFoundError:
        scene_data = _image_file_to_data_url(MODULE2_PART1_SCENE)

    matched_count = sum(1 for m in matches if m['matched'])
    return {
        'threshold': threshold,
        'scene': {
            'filename': MODULE2_PART1_SCENE.name,
            'image': scene_data,
        },
        'matches': matches,
        'summary': {
            'matched': matched_count,
            'total': len(matches),
        }
    }


def _load_module2_part3_references():
    """Return the static scene preview plus all template thumbnails."""
    if not MODULE2_PART1_SCENE.exists():
        raise FileNotFoundError(f"Scene file missing: {MODULE2_PART1_SCENE}")

    template_files = module2_part3.find_template_files(MODULE2_PART1_DIR, limit=MODULE2_PART3_TEMPLATE_LIMIT)
    if not template_files:
        raise FileNotFoundError(
            "No template files found in resources/m2 (expected files starting with 'template_')."
        )

    templates_payload = []
    for tpl_path in template_files:
        try:
            templates_payload.append({
                'filename': tpl_path.name,
                'label': tpl_path.stem,
                'image': _image_file_to_data_url(tpl_path),
            })
        except FileNotFoundError:
            continue

    return {
        'scene': {
            'filename': MODULE2_PART1_SCENE.name,
            'image': _image_path_to_preview_data_url(
                MODULE2_PART1_SCENE,
                MODULE2_PREVIEW_MAX_DIM,
                jpeg_quality=MODULE2_JPEG_QUALITY,
            ),
        },
        'templates': templates_payload,
    }


def _run_module2_part3(threshold: float, blur_multiplier: float):
    """Execute the CLI logic from module2_part3 via server-side call."""
    if not MODULE2_PART1_SCENE.exists():
        raise FileNotFoundError(f"Scene file missing: {MODULE2_PART1_SCENE}")

    template_files = module2_part3.find_template_files(MODULE2_PART1_DIR, limit=MODULE2_PART3_TEMPLATE_LIMIT)
    if not template_files:
        raise FileNotFoundError(
            "No template files found in resources/m2 (expected files starting with 'template_')."
        )

    scene = cv2.imread(str(MODULE2_PART1_SCENE))
    if scene is None:
        raise ValueError(f"Failed to load scene image: {MODULE2_PART1_SCENE}")
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    all_detections = []
    skipped_templates = []

    for tpl_path in template_files:
        tpl = cv2.imread(str(tpl_path))
        if tpl is None:
            skipped_templates.append({'filename': tpl_path.name, 'reason': 'unreadable'})
            continue
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        boxes, scores = module2_part3.match_template_once(scene_gray, tpl_gray, threshold=threshold)
        if not boxes:
            continue

        keep_idxs = module2_part1.non_max_suppression(boxes, scores, iou_threshold=0.35)
        for idx in keep_idxs:
            all_detections.append((boxes[idx], scores[idx], tpl_path.stem))

    drawn = module2_part3.draw_detections(scene, all_detections)
    boxes_to_blur = [det[0] for det in all_detections]
    blurred = module2_part3.blur_regions(drawn, boxes_to_blur, blur_multiplier=blur_multiplier)

    detection_payload = [
        {
            'label': label,
            'score': score,
            'box': {
                'x1': box[0],
                'y1': box[1],
                'x2': box[2],
                'y2': box[3],
            }
        }
        for (box, score, label) in all_detections
    ]

    scene_preview = _image_array_to_preview_data_url(
        scene,
        MODULE2_PREVIEW_MAX_DIM,
        jpeg_quality=MODULE2_JPEG_QUALITY,
    )
    detections_preview = _image_array_to_preview_data_url(
        drawn,
        MODULE2_PREVIEW_MAX_DIM,
        jpeg_quality=MODULE2_JPEG_QUALITY,
    )
    blurred_preview = _image_array_to_preview_data_url(
        blurred,
        MODULE2_PREVIEW_MAX_DIM,
        jpeg_quality=MODULE2_JPEG_QUALITY,
    )

    result = {
        'threshold': threshold,
        'blurMultiplier': blur_multiplier,
        'scene': {
            'filename': MODULE2_PART1_SCENE.name,
            'image': scene_preview,
        },
        'detectionsImage': detections_preview,
        'blurredImage': blurred_preview,
        'detections': detection_payload,
        'summary': {
            'detected': len(all_detections),
            'templatesTested': len(template_files),
            'skippedTemplates': skipped_templates,
        }
    }

    if all_detections:
        result['message'] = f"Detected {len(all_detections)} region(s) with threshold {threshold:.2f}."
    else:
        result['message'] = f"No objects detected at threshold {threshold:.2f}."

    return result

# --- Routes ---

@app.route('/')
def home():
    """Serves the main frontend HTML page."""
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    """Serve the favicon from the static folder at the root path /favicon.ico.

    Browsers usually request /favicon.ico — Flask serves static files under /static/, so
    we add this small route so the favicon is available at the expected path.
    """
    return send_from_directory(app.static_folder, 'favicon.ico')


@app.route('/instructions/<module_id>')
def module_instructions(module_id: str):
    """Serve the assignment PDF for the requested module, if it exists."""
    key = (module_id or '').lower()
    pdf_name = MODULE_INSTRUCTION_FILES.get(key)
    if not pdf_name:
        abort(404)

    pdf_path = INSTRUCTIONS_DIR / pdf_name
    if not pdf_path.exists():
        abort(404)

    return send_from_directory(INSTRUCTIONS_DIR, pdf_name)


@app.route('/written/<module_id>/<question_id>')
def written_answer(module_id: str, question_id: str):
    """Serve written-response PDFs for specific module/question combinations."""
    module_key = (module_id or '').lower()
    question_key = (question_id or '').lower()

    module_key = WRITTEN_MODULE_ALIASES.get(module_key, module_key)
    question_key = WRITTEN_QUESTION_ALIASES.get(question_key, question_key)

    module_answers = WRITTEN_ANSWER_FILES.get(module_key)
    if not module_answers:
        abort(404)

    pdf_name = module_answers.get(question_key)
    if not pdf_name:
        abort(404)

    pdf_path = WRITTEN_ANSWERS_DIR / pdf_name
    if not pdf_path.exists():
        abort(404)

    return send_file(pdf_path, mimetype='application/pdf')


@app.route('/media/m56/<path:filename>')
def serve_module56_media(filename):
    """Serve whitelisted Module 5 & 6 media assets (videos, mask archives)."""
    if not filename or '..' in filename:
        abort(404)
    safe_name = Path(filename).name
    target_path = MODULE56_DIR / safe_name
    if not target_path.exists():
        abort(404)
    mimetype = VIDEO_MIME_MAP.get(target_path.suffix.lower())
    return send_from_directory(str(MODULE56_DIR), safe_name, mimetype=mimetype)


def _parse_numeric(data, key, allow_zero=False):
    """Try to coerce JSON field `key` into float, raising ValueError on failure."""
    if key not in data:
        raise ValueError(f"Missing field: {key}")
    try:
        value = float(data[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be numeric")

    if not allow_zero and value == 0:
        raise ValueError(f"{key} must be non-zero")
    if value < 0:
        raise ValueError(f"{key} must be positive")
    return value


def _load_module1_samples():
    """Load sample images & measurement metadata for Module 1 from resources/m1/.
    Returns a dict with 'reference' and 'test' keys (filename, image data URL, measurements).
    """
    if not MODULE1_DIR.exists():
        raise FileNotFoundError(f"Module1 resources directory missing: {MODULE1_DIR}")

    ref_path = MODULE1_DIR / 'reference.jpg'
    test_path = MODULE1_DIR / 'test.jpg'
    md_path = MODULE1_DIR / 'measurements.md'

    if not ref_path.exists() or not test_path.exists():
        raise FileNotFoundError("Module 1 reference/test images are missing in resources/m1")

    measurements = {}
    if md_path.exists():
        # Very small markdown parser for the table in measurements.md
        try:
            text = md_path.read_text(encoding='utf-8')
            for line in text.splitlines():
                line = line.strip()
                if not line or not line.startswith('|'):
                    continue
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) != 3:
                    continue
                image, metric, value = parts
                # normalize key by stripping extension
                key = Path(image).name
                if key not in measurements:
                    measurements[key] = {}
                # unify metric names
                metric_key = metric.lower().replace(' ', '_')
                # strip 'cm', markdown '*' and any parenthetical notes; cast to float if possible
                val_str = value.replace('cm', '').replace('*', '').strip()
                if '(' in val_str:
                    val_str = val_str.split('(')[0].strip()
                try:
                    val = float(val_str)
                except Exception:
                    val = val_str
                # Normalize metric keys to easier names used in the UI
                if 'distance' in metric_key:
                    measurements[key]['distance'] = val
                elif 'width' in metric_key:
                    measurements[key]['width'] = val
                else:
                    measurements[key][metric_key] = val
        except Exception:
            pass

    return {
        'reference': {
            'filename': ref_path.name,
            'image': _image_file_to_data_url(ref_path),
            'realWidth': measurements.get('reference.jpg', {}).get('width'),
            'distance': measurements.get('reference.jpg', {}).get('distance')
        },
        'test': {
            'filename': test_path.name,
            'image': _image_file_to_data_url(test_path),
            'distance': measurements.get('test.jpg', {}).get('distance'),
            'expectedWidth': measurements.get('test.jpg', {}).get('width')
        }
    }


def _load_module7_part1_measurements():
    """Parse the Module 7 Part 1 measurements markdown (cm values only)."""
    if not MODULE7_PART1_MEASUREMENTS.exists():
        raise FileNotFoundError(f"Module 7 Part 1 measurements missing: {MODULE7_PART1_MEASUREMENTS}")

    text = MODULE7_PART1_MEASUREMENTS.read_text(encoding='utf-8')
    values = {
        'referenceDistanceCm': None,
        'referenceWidthCm': None,
        'baselineCm': None
    }

    def extract_cm(fragment: str):
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*cm', fragment)
        return float(match.group(1)) if match else None

    baseline_pending = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        value = extract_cm(lower)

        if 'distance from camera' in lower and value is not None:
            values['referenceDistanceCm'] = value
            baseline_pending = False
        elif 'width of object' in lower and value is not None:
            values['referenceWidthCm'] = value
            baseline_pending = False
        elif 'point difference' in lower or 'baseline' in lower:
            if value is not None:
                values['baselineCm'] = value
                baseline_pending = False
            else:
                baseline_pending = True
        elif baseline_pending and value is not None:
            values['baselineCm'] = value
            baseline_pending = False

        if all(v is not None for v in values.values()):
            break

    return values


def _load_module7_part1_samples():
    required = [MODULE7_PART1_REFERENCE, MODULE7_PART1_LEFT, MODULE7_PART1_RIGHT]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Module 7 Part 1 sample asset missing: {missing[0]}")

    try:
        measurements = _load_module7_part1_measurements()
    except FileNotFoundError:
        measurements = {}

    return {
        'reference': {
            'filename': MODULE7_PART1_REFERENCE.name,
            'image': _image_file_to_data_url(MODULE7_PART1_REFERENCE)
        },
        'left': {
            'filename': MODULE7_PART1_LEFT.name,
            'image': _image_file_to_data_url(MODULE7_PART1_LEFT)
        },
        'right': {
            'filename': MODULE7_PART1_RIGHT.name,
            'image': _image_file_to_data_url(MODULE7_PART1_RIGHT)
        },
        'measurements': measurements
    }


@app.route('/api/a1/focal-length', methods=['POST'])
def module1_calculate_focal_length():
    data = request.get_json(silent=True) or {}
    try:
        pixel_width = _parse_numeric(data, 'pixelWidth')
        real_width = _parse_numeric(data, 'realWidth')
        distance = _parse_numeric(data, 'distance')
        focal_length = calculate_focal_length(pixel_width, real_width, distance)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'focalLength': focal_length,
        'refPixelWidth': pixel_width,
        'refRealWidth': real_width,
        'refDistance': distance
    })


@app.route('/api/a1/samples', methods=['GET'])
def module1_samples():
    """Return sample reference & test images plus measurement defaults for Module 1.
    The images are returned as base64 data URLs to make it easy for the frontend to preview them.
    """
    try:
        data = _load_module1_samples()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to load Module 1 samples: {exc}'}), 500

    return jsonify(data)


@app.route('/api/a2/part2/sample', methods=['GET'])
def get_a2_part2_sample():
    """Return the pre-configured sample image as a data URL for the frontend preview."""
    try:
        if not MODULE2_PART2_SAMPLE.exists():
            raise FileNotFoundError(f"Sample image missing at {MODULE2_PART2_SAMPLE}")
        return jsonify({
            'filename': MODULE2_PART2_SAMPLE.name,
            'image': _image_file_to_data_url(MODULE2_PART2_SAMPLE),
        })
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to load sample image: {exc}'}), 500


@app.route('/api/a1/real-width', methods=['POST'])
def module1_calculate_real_width():
    data = request.get_json(silent=True) or {}
    try:
        pixel_width = _parse_numeric(data, 'pixelWidth')
        focal_length = _parse_numeric(data, 'focalLength')
        distance = _parse_numeric(data, 'distance')
        real_width = calculate_real_dimension(pixel_width, focal_length, distance)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'realWidth': real_width,
        'testPixelWidth': pixel_width,
        'testDistance': distance,
        'focalLength': focal_length
    })


@app.route('/api/a7/part1/sample', methods=['GET'])
def module7_part1_sample():
    try:
        data = _load_module7_part1_samples()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to load Module 7 Part 1 sample: {exc}'}), 500

    return jsonify(data)


@app.route('/api/a7/part1/focal-length', methods=['POST'])
def module7_part1_focal_length():
    data = request.get_json(silent=True) or {}
    try:
        pixel_width = _parse_numeric(data, 'pixelWidth')
        real_width = _parse_numeric(data, 'realWidth')
        distance = _parse_numeric(data, 'distance')
        focal_length = compute_focal_length_from_reference(pixel_width, real_width, distance)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'focalLength': focal_length,
        'refPixelWidth': pixel_width,
        'refRealWidth': real_width,
        'refDistance': distance
    })


@app.route('/api/a7/part1/depth', methods=['POST'])
def module7_part1_depth():
    data = request.get_json(silent=True) or {}
    try:
        focal_length = _parse_numeric(data, 'focalLength')
        baseline = _parse_numeric(data, 'baseline')
        disparity = _parse_numeric(data, 'disparity')
        depth_cm = compute_depth_from_stereo(focal_length, baseline, disparity)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'depthCm': depth_cm,
        'baselineCm': baseline,
        'disparityPx': disparity,
        'focalLength': focal_length
    })


@app.route('/api/a7/part1/measure', methods=['POST'])
def module7_part1_measure_segment():
    data = request.get_json(silent=True) or {}
    try:
        pixel_distance = _parse_numeric(data, 'pixelDistance')
        focal_length = _parse_numeric(data, 'focalLength')
        depth = _parse_numeric(data, 'depth')
        real_size = compute_real_size_from_depth(pixel_distance, focal_length, depth)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'realSizeCm': real_size,
        'pixelDistance': pixel_distance,
        'depthCm': depth,
        'focalLength': focal_length
    })


@app.route('/api/a56/part1/sample', methods=['GET'])
def get_a56_part1_sample():
    """Return the pre-configured sample video as a data URL for the frontend preview."""
    try:
        if not MODULE56_SAMPLE.exists():
            raise FileNotFoundError(f"Sample video missing at {MODULE56_SAMPLE}")
        data = MODULE56_SAMPLE.read_bytes()
        encoded = base64.b64encode(data).decode('ascii')
        mime = VIDEO_MIME_MAP.get(MODULE56_SAMPLE.suffix.lower(), 'application/octet-stream')
        return jsonify({'filename': MODULE56_SAMPLE.name, 'video': f"data:{mime};base64,{encoded}"})
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to load sample video: {exc}'}), 500


@app.route('/api/a7/part2/sample', methods=['GET'])
def get_a7_part2_sample():
    try:
        if not MODULE7_PART2_SAMPLE.exists():
            raise FileNotFoundError(f"Sample video missing at {MODULE7_PART2_SAMPLE}")
        return jsonify({'filename': MODULE7_PART2_SAMPLE.name, 'video': _video_file_to_data_url(MODULE7_PART2_SAMPLE)})
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to load Module 7 sample video: {exc}'}), 500


@app.route('/api/a56/part2/assets', methods=['GET'])
def get_a56_part2_assets():
    """Return metadata + bounding boxes for the fixed SAM2 showcase video."""
    try:
        payload = _load_module56_part2_assets()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 500
    except Exception as exc:
        return jsonify({'error': f'Failed to load Module 5/6 Part 2 assets: {exc}'}), 500

    response = dict(payload)
    response['videoUrl'] = url_for('serve_module56_media', filename=payload['videoFilename'])
    return jsonify(response)


@app.route('/api/a56/part1', methods=['POST'])
def handle_a56_part1_process():
    """Accept an uploaded video (multipart/form-data) or a `sample` form field and run the ArUco marker tracker.

    Returns a JSON containing a base64 data URL of the processed output video as `outputVideo`.
    """
    try:
        # multipart upload: input in request.files['video']
        video_file = None
        if request.files and 'video' in request.files:
            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({'error': 'Empty filename supplied'}), 400

        if video_file:
            filename = secure_filename(video_file.filename) or 'upload.mp4'
        else:
            # try JSON or form field sample
            payload = request.form or (request.get_json(silent=True) or {})
            sample_name = payload.get('sample') or (request.get_json(silent=True) or {}).get('sample')
            if not sample_name:
                return jsonify({'error': 'No video uploaded and no sample specified.'}), 400
            filename = Path(sample_name).name

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            return jsonify({'error': 'Unsupported file type'}), 400

        form_payload = request.form or (request.get_json(silent=True) or {})
        aruco_dict = form_payload.get('dict')
        padding = form_payload.get('padding', 1.0)
        try:
            padding = float(padding)
        except Exception:
            padding = 1.0
        if padding <= 0 or padding > 10:
            return jsonify({'error': 'Padding must be between 0 and 10'}), 400

        with TemporaryDirectory() as tmpdir:
            tmp_dir_path = Path(tmpdir)
            if video_file:
                input_path = tmp_dir_path / filename
                video_file.save(input_path)
            else:
                # sample requested
                requested_sample = Path(filename).name
                if requested_sample != MODULE56_SAMPLE.name or not MODULE56_SAMPLE.exists():
                    return jsonify({'error': 'Requested sample is not available'}), 400
                input_path = MODULE56_SAMPLE

            # Force output to MP4 for browser compatibility if possible
            output_path = tmp_dir_path / (Path(input_path).stem + '-tracked.mp4')

            try:
                module5_6.process_video(str(input_path), str(output_path), aruco_dict_name=aruco_dict or None, show_window=False, padding=padding)
            except Exception as exc:
                return jsonify({'error': f'Processing failed: {exc}'}), 500

            if not output_path.exists():
                return jsonify({'error': 'Processing failed to produce an output video.'}), 500

            output_path = _prepare_video_for_web(output_path)

            data = output_path.read_bytes()
            encoded = base64.b64encode(data).decode('ascii')
            mime = VIDEO_MIME_MAP.get(output_path.suffix.lower(), 'application/octet-stream')
            data_url = f"data:{mime};base64,{encoded}"

            return jsonify({
                'originalFilename': Path(filename).name,
                'outputFilename': output_path.name,
                'outputVideo': data_url,
                'padding': padding,
                'dict': aruco_dict or None,
                'message': f'Tracking complete for {Path(filename).name}'
            })
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Failed to run Module 5/6 Part 1: {exc}'}), 500


def _get_module4_files_from_request():
    files = request.files.getlist('images')
    if not files:
        files = request.files.getlist('images[]')
    return files


@app.route('/api/a4/samples', methods=['GET'])
def get_a4_samples():
    try:
        sample_paths = _list_module4_sample_paths()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404

    filenames = [path.name for path in sample_paths]
    if len(filenames) < MODULE4_MIN_IMAGES:
        return jsonify({'error': f"Example dataset must contain at least {MODULE4_MIN_IMAGES} portrait images."}), 400

    return jsonify({'count': len(filenames), 'filenames': filenames})


@app.route('/api/a4/part1', methods=['POST'])
def handle_a4_part1_upload():
    try:
        uploads, source = _collect_module4_inputs()
        panorama = _stitch_module4_part1([entry['image'] for entry in uploads])
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 422
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to stitch images with OpenCV pipeline: {exc}'}), 500

    source_label = 'example dataset' if source == 'sample' else 'uploaded set'
    return jsonify({
        'count': len(uploads),
        'filenames': [entry['filename'] for entry in uploads],
        'panorama': _image_array_to_data_url(panorama),
        'width': int(panorama.shape[1]),
        'height': int(panorama.shape[0]),
        'message': f"Stitched {len(uploads)} portrait images ({source_label}) using the OpenCV SIFT pipeline."
    })


@app.route('/api/a4/part2', methods=['POST'])
def handle_a4_part2_upload():
    try:
        uploads, source = _collect_module4_inputs()
        panorama = _stitch_module4_part2([entry['image'] for entry in uploads])
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 422
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to stitch images with custom pipeline: {exc}'}), 500

    source_label = 'example dataset' if source == 'sample' else 'uploaded set'
    return jsonify({
        'count': len(uploads),
        'filenames': [entry['filename'] for entry in uploads],
        'panorama': _image_array_to_data_url(panorama),
        'width': int(panorama.shape[1]),
        'height': int(panorama.shape[0]),
        'message': f"Stitched {len(uploads)} portrait images ({source_label}) using the scratch SIFT + affine pipeline."
    })


@app.route('/api/a4/part1/results', methods=['DELETE'])
def clear_a4_part1_results():
    """Stateless endpoint kept for parity with other modules."""
    return jsonify({'cleared': True})


@app.route('/api/a4/part2/results', methods=['DELETE'])
def clear_a4_part2_results():
    return jsonify({'cleared': True})


@app.route('/api/a2', methods=['POST'])
def handle_a2():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Module 2 code
    response = f"[Stub] Module 2 executed on input length: {len(user_input)}"
    return jsonify({'result': response})


@app.route('/api/a2/part1', methods=['POST'])
def handle_a2_part1_run():
    data = request.get_json(silent=True) or {}
    threshold = data.get('threshold', 0.8)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return jsonify({'error': 'Threshold must be a numeric value between 0.0 and 1.0.'}), 400

    if not 0.0 <= threshold <= 1.0:
        return jsonify({'error': 'Threshold must be between 0.0 and 1.0.'}), 400

    try:
        results = _run_module2_part1(threshold)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Failed to run template matching: {exc}'}), 500

    return jsonify(results)


@app.route('/api/a2/part1/scene', methods=['GET'])
def get_a2_part1_scene():
    if not MODULE2_PART1_SCENE.exists():
        return jsonify({'error': f"Scene file missing at {MODULE2_PART1_SCENE}"}), 404

    try:
        image_data = _image_path_to_preview_data_url(
            MODULE2_PART1_SCENE,
            MODULE2_PREVIEW_MAX_DIM,
            jpeg_quality=MODULE2_JPEG_QUALITY,
        )
    except FileNotFoundError:
        image_data = _image_file_to_data_url(MODULE2_PART1_SCENE)

    return jsonify({
        'filename': MODULE2_PART1_SCENE.name,
        'image': image_data,
    })


@app.route('/api/a2/part1/results', methods=['DELETE', 'POST'])
def clear_a2_part1_results():
    deleted = _clear_module2_part1_results()
    return jsonify({'deleted': deleted})


@app.route('/api/a2/part2', methods=['POST'])
def handle_a2_part2_process():
    # Support either an uploaded file (multipart/form-data) or a request to use
    # a pre-configured sample image (form field 'sample' or JSON 'sample').
    image_file = None
    sample_name = None

    if request.files and 'image' in request.files:
        image_file = request.files['image']
    else:
        # try reading a form field (multipart or standard form)
        sample_name = (request.form.get('sample') or
                       (request.get_json(silent=True) or {}).get('sample'))
    # If using uploaded image, ensure a filename is present
    if image_file and image_file.filename == '':
        return jsonify({'error': 'Empty filename supplied'}), 400

    if image_file:
        filename = secure_filename(image_file.filename) or 'upload.png'
    else:
        # if we will use a sample, set the filename accordingly for metadata
        filename = MODULE2_PART2_SAMPLE.name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        with TemporaryDirectory() as tmpdir:
            tmp_dir_path = Path(tmpdir)
            if image_file:
                input_path = tmp_dir_path / filename
                image_file.save(input_path)
            else:
                # If sample requested, construct a path to the preconfigured sample
                # Default to MODULE2_PART2_SAMPLE (tree.jpg) if present, otherwise error
                if not sample_name:
                    return jsonify({'error': 'No image file provided and no sample specified'}), 400
                # for now only 'tree.jpg' is supported; map sample name to path defensively
                requested_sample = Path(sample_name).name
                if requested_sample != MODULE2_PART2_SAMPLE.name or not MODULE2_PART2_SAMPLE.exists():
                    return jsonify({'error': 'Requested sample is not available'}), 400
                input_path = MODULE2_PART2_SAMPLE

            blur_override = tmp_dir_path / f"{input_path.stem}_blurred.png"
            restored_override = tmp_dir_path / f"{input_path.stem}_restored.png"

            (
                blur_path,
                restored_path,
                kernel_used,
                sigma_used,
            ) = module2_process_image(
                input_image=input_path,
                kernel_size=None,
                sigma=None,
                balance=1e-2,
                eps=1e-6,
                blur_output=str(blur_override),
                restored_output=str(restored_override),
            )

            def _path_to_data_url(image_path: Path) -> str:
                data = image_path.read_bytes()
                encoded = base64.b64encode(data).decode('ascii')
                return f"data:image/png;base64,{encoded}"

            return jsonify({
                'blurredImage': _path_to_data_url(Path(blur_path)),
                'restoredImage': _path_to_data_url(Path(restored_path)),
                'originalFilename': filename,
                'kernelSize': kernel_used,
                'sigma': sigma_used,
            })
    except FileNotFoundError:
        return jsonify({'error': 'Uploaded image could not be processed'}), 400
    except Exception as exc:
        return jsonify({'error': f'Processing failed: {exc}'}), 500


@app.route('/api/a2/part3/references', methods=['GET'])
def get_a2_part3_references():
    try:
        refs = _load_module2_part3_references()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to load references: {exc}'}), 500

    return jsonify(refs)


@app.route('/api/a2/part3', methods=['POST'])
def handle_a2_part3_run():
    data = request.get_json(silent=True) or {}

    threshold = data.get('threshold', 0.75)
    blur_multiplier = data.get('blurMultiplier', 2.5)

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return jsonify({'error': 'Threshold must be a numeric value between 0.0 and 1.0.'}), 400

    try:
        blur_multiplier = float(blur_multiplier)
    except (TypeError, ValueError):
        return jsonify({'error': 'Blur multiplier must be a positive numeric value.'}), 400

    if not 0.0 <= threshold <= 1.0:
        return jsonify({'error': 'Threshold must be between 0.0 and 1.0.'}), 400
    if blur_multiplier <= 0 or blur_multiplier > 25:
        return jsonify({'error': 'Blur multiplier must be between 0 and 25.'}), 400

    try:
        results = _run_module2_part3(threshold, blur_multiplier)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except Exception as exc:
        return jsonify({'error': f'Failed to run Module 2 Part 3: {exc}'}), 500

    return jsonify(results)

@app.route('/api/a3/samples', methods=['GET'])
def get_a3_samples():
    try:
        sample_paths = _list_module3_sample_paths()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'count': len(sample_paths),
        'filenames': [path.name for path in sample_paths],
    })


@app.route('/api/a3/part4/samples', methods=['GET'])
def get_a3_part4_samples():
    try:
        sample_paths = _list_module3_part4_sample_paths()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    return jsonify({
        'count': len(sample_paths),
        'filenames': [path.name for path in sample_paths],
    })


@app.route('/api/a3/part1', methods=['POST'])
def handle_a3_part1():
    try:
        uploads, source = _collect_module3_inputs()
        results = _run_module3_processor(uploads, module3_part1.process_gradients_log, 'Part 1')
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        return jsonify({'error': f'Failed to run Module 3 Part 1: {exc}'}), 500

    source_label = 'example dataset' if source == 'sample' else 'uploaded set'
    count = len(results)
    return jsonify({
        'count': count,
        'source': source,
        'results': results,
        'message': f"Generated gradient magnitude, angle, and Laplacian-of-Gaussian grids for {count} images using the {source_label}."
    })


@app.route('/api/a3/part2', methods=['POST'])
def handle_a3_part2():
    try:
        uploads, source = _collect_module3_inputs()
        results = _run_module3_processor(uploads, module3_part1.detect_features, 'Part 2')
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        return jsonify({'error': f'Failed to run Module 3 Part 2: {exc}'}), 500

    source_label = 'example dataset' if source == 'sample' else 'uploaded set'
    return jsonify({
        'count': len(results),
        'source': source,
        'results': results,
        'message': f"Detected edge (green) and corner (red) keypoints across {len(results)} images using the {source_label}."
    })


@app.route('/api/a3/part3', methods=['POST'])
def handle_a3_part3():
    try:
        uploads, source = _collect_module3_inputs()
        results = _run_module3_processor(
            uploads,
            module3_part1.find_exact_boundary,
            'Part 3',
            max_process_dim=MODULE3_PART3_PROCESS_MAX_DIM,
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        return jsonify({'error': f'Failed to run Module 3 Part 3: {exc}'}), 500

    source_label = 'example dataset' if source == 'sample' else 'uploaded set'
    return jsonify({
        'count': len(results),
        'source': source,
        'results': results,
        'message': f"Outlined the dominant object boundaries for {len(results)} images using the {source_label}."
    })


@app.route('/api/a3/part4', methods=['POST'])
def handle_a3_part4():
    iterations = 5
    expansion = 0.15

    payload = _get_request_payload()
    if payload:
        try:
            iterations = int(float(payload.get('iterations', iterations)))
        except (TypeError, ValueError):
            iterations = 5
        try:
            expansion = float(payload.get('hullExpansion', expansion))
        except (TypeError, ValueError):
            expansion = 0.15

    iterations = max(1, min(iterations, 10))
    expansion = max(0.0, min(expansion, 0.5))

    try:
        uploads, source = _collect_module3_part4_inputs()
        results = _run_module3_part4_processor(uploads, iterations=iterations, expansion=expansion)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        return jsonify({'error': f'Failed to run Module 3 Part 4: {exc}'}), 500

    marker_counts = [entry.get('markerCount', 0) for entry in results]
    avg_markers = round(sum(marker_counts) / len(marker_counts), 2) if marker_counts else 0
    source_label = 'example dataset' if source == 'sample' else 'uploaded set'
    return jsonify({
        'count': len(results),
        'source': source,
        'results': results,
        'summary': {
            'iterations': iterations,
            'hullExpansion': expansion,
            'averageMarkers': avg_markers,
        },
        'message': f"Segmented {len(results)} images by anchoring GrabCut around the ArUco markers from the {source_label}."
    })


@app.route('/api/a3/part5/gallery', methods=['GET'])
def get_a3_part5_gallery():
    try:
        pairs = _list_module3_part5_pairs()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    results = []
    for original_path, sam_path in pairs:
        try:
            original_image = _image_file_to_data_url(original_path)
            sam_image = _image_file_to_data_url(sam_path)
        except Exception as exc:
            return jsonify({'error': f'Failed to load showcase image: {exc}'}), 500

        results.append({
            'filename': original_path.name,
            'originalImage': original_image,
            'samImage': sam_image,
        })

    metadata = {
        'model': 'sam2.1_hiera_large',
        'device': 'RTX 4090',
        'command': (
            'python module3_part3.py <path/to/image-folder> '
            '--model-config <path/to/sam2.1_hiera_l.yaml> '
            '--checkpoint <path/to/sam2.1_hiera_large.pt> '
            '--device [auto|cuda|mps|cpu] --verbose'
        ),
        'pipPackages': [
            'torch',
            'sam2',
            'hydra-core',
            'omegaconf',
            'opencv-python',
            'numpy'
        ],
    }

    return jsonify({
        'count': len(results),
        'results': results,
        'metadata': metadata,
        'message': 'Offline ArUco GrabCut vs. SAM2 comparisons generated with the sam2.1_hiera_large checkpoint on an RTX 4090.'
    })


@app.route('/api/a3/part5/raw-download', methods=['GET'])
def download_a3_part5_raw():
    try:
        sample_paths = _list_module3_part4_sample_paths()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sample_paths:
            archive.write(path, arcname=path.name)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='module3-part5-raw-images.zip'
    )


@app.route('/api/a3', methods=['POST'])
def handle_a3():
    data = request.json or {}
    user_input = data.get('input', '')

    response = f"Module 3 UI has dedicated Part 1-3 endpoints. Payload echo: {user_input[:64]}"
    return jsonify({'result': response})

@app.route('/api/a4', methods=['POST'])
def handle_a4():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Module 4 code
    response = f"[Stub] Module 4 logic placeholder."
    return jsonify({'result': response})

@app.route('/api/a56', methods=['POST'])
def handle_a56():
    data = request.json
    user_input = data.get('input', '')
    # Legacy stub route: inform client to use the part-specific API (Part 1 - tracker)
    response = "Module 5 & 6 endpoint is deprecated. Use '/api/a56/part1' to upload videos for the tracker."
    return jsonify({'result': response})

@app.route('/api/a7', methods=['POST'])
def handle_a7():
    data = request.json
    user_input = data.get('input', '')
    
    # TODO: Connect Module 7 code
    response = f"[Stub] Module 7 final project placeholder."
    return jsonify({'result': response})


@app.route('/api/a7/part2', methods=['POST'])
def handle_a7_part2_process():
    """Accept a video upload or use the bundled sample to run pose + hand tracking."""
    try:
        payload = _get_request_payload()
        video_file = request.files.get('video') if request.files else None
        sample_name = payload.get('sample') if payload else None

        if video_file and video_file.filename == '':
            return jsonify({'error': 'Empty filename supplied.'}), 400

        if not video_file and not sample_name:
            return jsonify({'error': 'Please upload a video or choose the example video.'}), 400

        if video_file:
            filename = secure_filename(video_file.filename) or 'pose-tracking.mp4'
        else:
            filename = Path(sample_name).name

        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            return jsonify({'error': 'Unsupported video format.'}), 400

        with TemporaryDirectory() as tmpdir:
            tmp_dir_path = Path(tmpdir)
            input_path = tmp_dir_path / filename
            if video_file:
                video_file.save(input_path)
            else:
                if filename != MODULE7_PART2_SAMPLE.name or not MODULE7_PART2_SAMPLE.exists():
                    return jsonify({'error': 'Requested sample is not available.'}), 400
                shutil.copy2(MODULE7_PART2_SAMPLE, input_path)

            annotated_path = tmp_dir_path / f"{input_path.stem}_annotated.mp4"
            try:
                result = module7_part2.run_pose_and_hand_tracking(
                    input_path,
                    display=False,
                    annotated_output_path=annotated_path,
                )
            except Exception as exc:
                return jsonify({'error': f'Pose estimation failed: {exc}'}), 500

            csv_path = Path(result.get('csv_path')) if result.get('csv_path') else None
            if not csv_path or not csv_path.exists():
                return jsonify({'error': 'Pose estimation did not produce a CSV file.'}), 500

            annotated_file = result.get('annotated_video_path')
            annotated_preview = None
            annotated_filename = None
            if annotated_file:
                annotated_file = _prepare_video_for_web(Path(annotated_file))
                annotated_preview = _video_file_to_data_url(annotated_file)
                annotated_filename = annotated_file.name

            original_preview = _video_file_to_data_url(input_path)
            csv_data_url = _csv_to_data_url(csv_path)
            columns, preview_rows, row_count = _build_csv_preview(csv_path)

            summary = {
                'frameCount': result.get('frame_count'),
                'recordCount': result.get('record_count'),
                'fps': result.get('fps'),
                'durationSeconds': result.get('duration_seconds'),
                'frameWidth': result.get('frame_width'),
                'frameHeight': result.get('frame_height'),
                'csvRowCount': row_count,
            }

            response = {
                'message': f'Pose estimation complete for {filename}',
                'original': {
                    'filename': filename,
                    'video': original_preview,
                },
                'annotated': None,
                'csv': {
                    'filename': csv_path.name,
                    'dataUrl': csv_data_url,
                    'preview': {
                        'columns': columns,
                        'rows': preview_rows,
                    },
                    'rowCount': row_count,
                },
                'summary': summary,
            }
            if annotated_preview and annotated_filename:
                response['annotated'] = {
                    'filename': annotated_filename,
                    'video': annotated_preview,
                }

            return jsonify(response)
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Failed to run Module 7 Part 2: {exc}'}), 500


@app.route('/source/<path:filename>')
def serve_source(filename):
    """Serve raw source files from the hwsources/ directory.

    This implements a small, guarded endpoint so client-side UI can fetch
    Python source (or other text files) dynamically. It strictly prevents
    path traversal and only serves files with a safe extension.
    """
    # guard against directory traversal
    if filename.startswith('/') or '..' in filename:
        abort(404)

    # restrict to safe file types
    allowed_ext = {'.py', '.txt', '.md'}
    ext = None
    try:
        import os
        _, ext = os.path.splitext(filename)
    except Exception:
        abort(404)

    if ext not in allowed_ext:
        abort(404)

    # deliver the file contents from the hwsources directory
    try:
        # If the query contains a truthy `download` flag, serve the file as an attachment
        download_flag = request.args.get('download', '').lower()
        as_attachment = download_flag in ('1', 'true', 'yes', 'on')
        # Flask's send_from_directory supports `as_attachment` which will set
        # the Content-Disposition header and prompt a file download in the
        # user's browser. Still, the UI also uses an HTML anchor with the
        # download attribute for a direct client-side fallback.
        return send_from_directory('hwsources', filename, mimetype='text/plain', as_attachment=as_attachment)
    except Exception:
        abort(404)

if __name__ == '__main__':
    # Running on 0.0.0.0 ensures it's accessible externally on your Azure VM
    app.run(host='0.0.0.0', port=5000, debug=True)