# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def mse(i1: np.ndarray, i2: np.ndarray) -> float:
    return np.mean((i1 - i2) ** 2)

def psnr(i1: np.ndarray, i2: np.ndarray, max_pixel: float = 1.0) -> float:
    mse_val = mse(i1, i2)
    if mse_val == 0:
        return float('inf')  
    return 10 * np.log10((max_pixel ** 2) / mse_val)

def ssim(i1: np.ndarray, i2: np.ndarray, window_size: int = 11, k1: float = 0.01, k2: float = 0.03, L: float = 1.0) -> float:
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)
    
    sigma1_sq = np.var(i1)
    sigma2_sq = np.var(i2)
    sigma12 = np.cov(i1.flatten(), i2.flatten())[0, 1]
    
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    
    return numerator / denominator

def npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    i1_flat = i1.flatten()
    i2_flat = i2.flatten()
    
    mean1 = np.mean(i1_flat)
    mean2 = np.mean(i2_flat)
    
    numerator = np.sum((i1_flat - mean1) * (i2_flat - mean2))
    denominator1 = np.sqrt(np.sum((i1_flat - mean1) ** 2))
    denominator2 = np.sqrt(np.sum((i2_flat - mean2) ** 2))
    
    if denominator1 == 0 or denominator2 == 0:
        return 0.0
    
    return numerator / (denominator1 * denominator2)

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    if i1.shape != i2.shape:
        raise ValueError("Tamanho incompatível")
    
    mse_val = mse(i1, i2)
    psnr_val = psnr(i1, i2)
    ssim_val = ssim(i1, i2)
    npcc_val = npcc(i1, i2)
    
    return {
        "mse": mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "npcc": npcc_val
    }