"""
CSC 8830 Computer Vision
Dr. Ashwin Ashok
Avyuktkrishna Ramasamy
Module 2 Assignment Part 2 - Gaussian Blur and Inverse Filtering (Fourier Transform)

The purpose of the script is to apply Gaussian 
blur to an input image, then use Fourier Transform 
in order to unblur the image.

Process
1. Upon request, the user provides the path to a specific image
2. If the path is valid (file exists) and the file is valid then image is blurred
	a. Gaussian blur - kernel size and sigma are determined automatically by using image dimensions when not specified 
	(output is called <input>_b.<ext>) and restored (output is called <input>_restored.<ext>)
	b. Transforming blurred image and Gaussian kernel to frequency domain using FFT, applying inverse filter (dividing blurred image FFT by kernel FFT), and transforming back to spatial domain
3. Output images are saved to disk
4. Kernel size and sigma values, along with image file paths are displayed to the user

Usage:
    Run the script from CLI - python module2_part2.py
    Enter the image to the path - /path/to/the/image.jpg
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

# Global constants
AUTO_KERNEL_MIN = 5 # Minimum kernel size
AUTO_KERNEL_MAX = 151 # Maximum kernel size
AUTO_KERNEL_FRACTION =  0.03 # Scale woukd be relative to the smaller image dimension

# Creating the Gaussian kernel
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
	# Kernel size must be odd number
	if size % 2 == 0: raise ValueError("Gaussian kernel size must be odd")
	ax = np.arange(size) - size // 2
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
	kernel /= np.sum(kernel)
	return kernel.astype(np.float32)

# Using OpenCV Gaussian Blur function
def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
	return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

# If user doesn't specify parameters, calculate the parameters such that it will scale with the image resolution
def _resolve_kernel_params(shape: Tuple[int, int, int],	kernel_size: Optional[int],	sigma: Optional[float]) -> Tuple[int, float]:
	if kernel_size is None or kernel_size <= 0:
		height, width = shape[:2]
		minimum_dimension = min(height, width)
		suggested = max(AUTO_KERNEL_MIN, int(round(minimum_dimension * AUTO_KERNEL_FRACTION)))
		if suggested % 2 == 0: suggested += 1
		kernel_size = min(suggested, AUTO_KERNEL_MAX)
	if sigma is None or sigma <= 0: sigma = max(1.0, kernel_size / 6.0)
	return int(kernel_size), float(sigma)

# Calculating the fast Fourier transform
def _kernel_fft(shape: Tuple[int, int], kernel: np.ndarray) -> np.ndarray:
	return np.fft.fft2(np.fft.ifftshift(kernel), s=shape)

# Filtering in frequency domain (easier than spatial domain)
def inverse_filter_deblur(blurred: np.ndarray, kernel_size: int, sigma: float, balance: float = 1e-2, eps: float = 1e-6) -> np.ndarray:
	# Gaussian kernel is instantiated
	kernel = gaussian_kernel(kernel_size, sigma)
	h, w = blurred.shape[:2]
	# Switching kernel to frequency domain
	kernel_fft = _kernel_fft((h, w), kernel)
	kernel_power = np.abs(kernel_fft) ** 2
	conj_kernel = np.conj(kernel_fft)
	# Blurred image is transformed to frequency domain
	blurred_fft = np.fft.fft2(blurred, axes=(0, 1))
	# Inverse filter applied in frequency domain
	restored_fft = (conj_kernel[..., None] * blurred_fft) / (kernel_power[..., None] + balance + eps)
	# Transformation back to spatial domain
	restored = np.real(np.fft.ifft2(restored_fft, axes=(0, 1)))
	return np.clip(restored, 0.0, 1.0)

# Load the image
def load_image(path: Path) -> np.ndarray:
	# Load image as float32 in [0, 1] range
	image = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if image is None:
		raise FileNotFoundError(f"The image was not found :- {path}")
	return image.astype(np.float32) / 255.0

# Save the image
def save_image(path: Path, image: np.ndarray) -> None:
	# Persist float image array as uint8 BGR
	path.parent.mkdir(parents=True, exist_ok=True)
	image_to_save = np.clip(image * 255.0, 0, 255).astype(np.uint8)
	cv2.imwrite(str(path), image_to_save)

# Create the file names/paths for output images
def derive_output_paths(input_path: Path, blur_override: Optional[str],	restored_override: Optional[str]) -> Tuple[Path, Path]:
	stem = input_path.stem
	suffix = input_path.suffix or ".png"
	blur_path = Path(blur_override) if blur_override else input_path.with_name(f"{stem}_b{suffix}")
	restored_path = (
		Path(restored_override)
		if restored_override
		else input_path.with_name(f"{stem}_restored{suffix}")
	)
	return blur_path, restored_path

# Image processing
def process_image(input_image: Path, kernel_size: Optional[int], sigma: Optional[float], balance: float, eps: float, blur_output: Optional[str], restored_output: Optional[str]) -> Tuple[Path, Path, int, float]:
	original = load_image(input_image)
	resolved_kernel, resolved_sigma = _resolve_kernel_params(original.shape, kernel_size, sigma)
	blurred = apply_gaussian_blur(original, resolved_kernel, resolved_sigma)
	restored = inverse_filter_deblur(blurred, resolved_kernel, resolved_sigma, balance, eps)
	blur_path, restored_path = derive_output_paths(input_image, blur_output, restored_output)
	save_image(blur_path, blurred)
	save_image(restored_path, restored)
	return blur_path, restored_path, resolved_kernel, resolved_sigma

# Main function
def main() -> None:
	# Prompt user for image path
	image_path_str = input("Enter the file path for the image :- ").strip()
	input_path = Path(image_path_str)
	
	# Using some default parameters for Gaussian blur and inverse filtering
	blur_path, restored_path, kernel_used, sigma_used = process_image(input_image=input_path, kernel_size=None, sigma=None, balance=1e-2, eps=1e-6, blur_output=None, restored_output=None)
	print(f"Blurred image saved to {blur_path}")
	print(f"Restored image saved to {restored_path}")
	print(f"Parameters used -> kernel size: {kernel_used}, sigma: {sigma_used:.2f}")

if __name__ == "__main__":
	main()