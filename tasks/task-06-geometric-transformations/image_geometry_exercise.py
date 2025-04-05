# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def translate_image(img: np.ndarray, dx: int = 10, dy: int = 10) -> np.ndarray:
    translated = np.zeros_like(img)
    translated[dy:, dx:] = img[:-dy, :-dx]
    return translated

def rotate_image(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=-1)

def stretch_image(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    h, w = img.shape
    new_w = int(w * scale)
    
    # Create coordinate arrays
    x = np.linspace(0, w-1, new_w)
    y = np.arange(h)
    
    # Interpolate using nearest neighbor for simplicity
    x_idx = np.clip(np.round(x), 0, w-1).astype(int)
    stretched = img[np.ix_(y, x_idx)]
    
    return stretched

def mirror_image(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1]

def distort_image(img: np.ndarray, strength: float = 0.1) -> np.ndarray:
    h, w = img.shape
    y, x = np.indices((h, w))
    
    # Normalize coordinates to [-1, 1]
    x_norm = 2 * (x - w//2) / w
    y_norm = 2 * (y - h//2) / h
    
    # Calculate radial distance
    r = np.sqrt(x_norm**2 + y_norm**2)
    
    # Apply distortion (simple barrel distortion)
    distortion = 1 + strength * r**2
    x_dist = x_norm * distortion
    y_dist = y_norm * distortion
    
    # Convert back to original coordinates
    x_new = (x_dist * w/2 + w//2).clip(0, w-1)
    y_new = (y_dist * h/2 + h//2).clip(0, h-1)
    
    # Interpolate using nearest neighbor
    x_idx = np.round(x_new).astype(int)
    y_idx = np.round(y_new).astype(int)
    
    return img[y_idx, x_idx]

def apply_geometric_transformations(img: np.ndarray) -> dict:
    return {
        "translated": translate_image(img),
        "rotated": rotate_image(img),
        "stretched": stretch_image(img),
        "mirrored": mirror_image(img),
        "distorted": distort_image(img)
    }