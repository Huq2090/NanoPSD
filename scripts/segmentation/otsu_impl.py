"""
Implementation of the BaseSegmenter interface using classical Otsu thresholding.
This wraps the existing `segment_particles` function so it can be used
interchangeably with future AI-based segmenters.
"""

from .base import BaseSegmenter
from .otsu_segment import segment_particles


class OtsuSegmenter(BaseSegmenter):
    def __init__(self, min_size=3, max_size=None):
        """
        Parameters
        ----------
        min_size : int
            Minimum object size (in pixels) to keep after segmentation.

        max_size : int, optional
            Maximum object size (in pixels) to keep after segmentation.
            If None, no maximum filter is applied.
        """
        self.min_size = min_size
        self.max_size = max_size

    def segment(self, binary_image):
        """
        Apply Otsu-based segmentation (connected component labeling).

        Parameters
        ----------
        binary_image : np.ndarray (bool)
            Binary mask (True = object, False = background).

        Returns
        -------
        labeled : np.ndarray (int)
            Labeled mask (0 = background, 1..N = object id).
        regions : list of skimage.measure._regionprops.RegionProperties
            Region properties for each segmented object.
        """
        return segment_particles(
            binary_image, min_size=self.min_size, max_size=self.max_size
        )
