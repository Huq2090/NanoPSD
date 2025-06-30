# Import OpenCV for image processing functions
import cv2

def preprocess_image(image_path):
    """
    Preprocesses a microscopy image by enhancing contrast, smoothing, and thresholding.
    
    Parameters:
    -----------
    image_path : str
        Path to the grayscale input image.
    
    Returns:
    --------
    binary : np.ndarray (bool)
        A binary image (True for foreground/particles, False for background).
    image : np.ndarray (uint8)
        The original grayscale image.
    """
    
    # Step 1: Read the image in grayscale mode
    # Grayscale simplifies analysis and is appropriate for microscopy images
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Create a CLAHE (Contrast Limited Adaptive Histogram Equalization) object
    # clipLimit controls contrast enhancement (higher = more enhancement)
    # tileGridSize defines the number of regions CLAHE is applied to (8x8 tiles here)
    # A typical value of clipLimit is between 2.0 and 4.0. The image is divided into 8×8 tiles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Step 3: Apply CLAHE to enhance local contrast
    # This helps make dim particles more distinguishable from the background
    enhanced = clahe.apply(image)

    # Step 4: Apply Gaussian blur to smooth the image
    # This smooths out small fluctuations or roughness
    # Kernel size (3, 3) helps reduce small noise without significantly affecting particle edges
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Step 5: Apply Otsu's thresholding to automatically binarize the image
    # This automatically determines the optimal threshold value to separate foreground and background
    # Result is a binary image: pixels > threshold become 255 (white), else 0 (black)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Invert the binary image
    # Otsu thresholding often yields white background and dark particles
    # We invert it so that particles are white (foreground = 255) and background is black (0)
    binary = 255 - binary

    # Step 7: Return the binary image as a boolean array (True = foreground particle)
    # and the original grayscale image (used for visualization or further analysis)
    return binary > 0, image


