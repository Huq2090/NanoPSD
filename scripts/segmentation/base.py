"""
Base interface for all segmentation methods.
This ensures that both Classical (Otsu) and AI-based segmenters
share the same function signature and can be swapped easily.
"""
from abc import ABC, abstractmethod

class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, image_or_binary):
        """
        Perform segmentation on an image or binary mask.

        Parameters
        ----------
        image_or_binary : np.ndarray
            Input to the segmentation method. Depending on the implementation,
            this could be a grayscale image (AI) or a binary image (Otsu).

        Returns
        -------
        labeled : np.ndarray (int)
            Labeled mask (0 = background, 1..N = object id).
        regions : list of skimage.measure._regionprops.RegionProperties
            Properties for each segmented object (area, centroid, etc.).
        """
        raise NotImplementedError