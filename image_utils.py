import cv2
import numpy as np
from scipy.ndimage import rotate
from scipy.fftpack import fft

def preprocess_input_image(image_path):
    """Load, preprocess, and normalize the image."""
    SIZE = 224  # Image size expected by the model
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Error: Image could not be loaded.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Maintain aspect ratio while resizing
    s = max(img.shape[:2])
    f = np.zeros((s, s, 3), np.uint8)
    ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2
    f[ay:ay + img.shape[0], ax:ax + img.shape[1]] = img
    img = cv2.resize(f, (SIZE, SIZE))

    # Normalize
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img

def compute_sharpness(img):
    """Compute image sharpness using the Laplacian variance."""
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def compute_brightness(img):
    """Compute image brightness (average pixel intensity)."""
    return np.mean(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))


def compute_contrast(img):
    """Compute image contrast (standard deviation of pixel values)."""
    return np.std(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))


def compute_fourier_descriptors(img):
    """Compute Fourier descriptors for frequency domain representation."""
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    f_transform = np.abs(fft(gray.flatten()))[:10]  # First 10 frequency components
    return (f_transform / np.max(f_transform)).tolist() if np.max(f_transform)!= 0 else f_transform.tolist()


def compute_image_metrics(img):
    """
    Compute sharpness, brightness, contrast, and Fourier descriptors.

    Args:
        img (numpy.ndarray): Input image as a NumPy array.

    Returns:
        dict: Dictionary containing the computed image metrics.
    """
    img = np.squeeze(img)  # Remove batch dimension
    return {
        "sharpness": compute_sharpness(img),
        "brightness": compute_brightness(img),
        "contrast": compute_contrast(img),
        "fourier_descriptors": compute_fourier_descriptors(img)
    }


def add_noise(image, noise_factor=0.02):
    """Add random Gaussian noise to the image."""
    # Ensure the image doesn't have batch dimension
    if len(image.shape) == 4:  # If shape is (batch, height, width, channels)
        image = np.squeeze(image, axis=0)
        
    noise = np.random.normal(0, noise_factor, image.shape)
    noisy_image = np.clip(image + noise, 0, 1)  # Keep values in range
    return noisy_image


def apply_blur(image, kernel_size=3):
    """Apply slight Gaussian blur to the image."""
    # Ensure the image doesn't have batch dimension
    if len(image.shape) == 4:  # If shape is (batch, height, width, channels)
        image = np.squeeze(image, axis=0)
    
    # Convert from float [0-1] to uint8 [0-255] for OpenCV
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Apply blur
    blurred = cv2.GaussianBlur(image_uint8, (kernel_size, kernel_size), 0)
    
    # Convert back to float [0-1]
    return blurred.astype(np.float32) / 255.0


def augment_image(image):
    """Apply all augmentations: noise and blur."""
    # Ensure the image doesn't have batch dimension
    if len(image.shape) == 4:  # If shape is (batch, height, width, channels)
        image = np.squeeze(image, axis=0)
        
    img = add_noise(image)
    img = apply_blur(img)
    
    # Add batch dimension back for prediction
    img = np.expand_dims(img, axis=0)
    return img