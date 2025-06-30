def segment_particles(binary_image, min_size=3):
    cleaned = morphology.remove_small_objects(binary_image, min_size=min_size)
    filled = ndimage.binary_fill_holes(cleaned)
    labeled, _ = ndimage.label(filled)
    regions = measure.regionprops(labeled)
    return labeled, regions



from skimage import measure, morphology  # Import particle measurement and morphological tools from scikit-image
from scipy import ndimage  # Import functions for binary image manipulation from SciPy

def segment_particles(binary_image, min_size=3):
    """
    Segments particles in a binary image using connected component analysis,
    removes small noise objects, fills internal holes, and returns labeled regions.

    Parameters:
    -----------
    binary_image : np.ndarray (bool)
        Binary mask of the image where True = foreground (particle), False = background.
    min_size : int, optional (default=3)
        Minimum size (in pixels) for objects to be retained. Smaller regions are considered noise.

    Returns:
    --------
    labeled : np.ndarray (int)
        Labeled image where each connected component (particle) has a unique positive integer label.
        Background is labeled as 0.
    regions : list of skimage.measure._regionprops.RegionProperties
        A list of region properties for each labeled particle, which can be used for size, shape, etc.
    """
    
    # Step 1: Remove small objects from the binary image
    # This is a morphological operation to eliminate noise and tiny specks
    # Any connected group of True pixels smaller than min_size is removed
    cleaned = morphology.remove_small_objects(binary_image, min_size=min_size)

    # Step 2: Fill internal holes within objects (particles)
    # Ensures that ring-shaped particles or porous features are treated as solid
    # For example, a particle shaped like a donut will be filled in
    filled = ndimage.binary_fill_holes(cleaned)

    # Step 3: Label connected components in the binary image
    # Each group of connected True pixels gets a unique label: 1, 2, 3, ..., N
    # Background pixels (False) are labeled as 0
    # `num_particles` counts the number of detected objects
    labeled, num_particles = ndimage.label(filled)

    # Step 4: Compute region properties of the labeled image
    # This returns measurements like area, centroid, bounding box, etc., for each particle
    regions = measure.regionprops(labeled)

    # Step 5: Return the labeled image and list of particle properties
    return labeled, regions

