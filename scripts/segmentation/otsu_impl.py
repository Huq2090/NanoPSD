# SPDX-License-Identifier: GPL-3.0-or-later
#
# NanoPSD: Automated Nanoparticle Shape Distribution Analysis
# Copyright (C) 2026 Md Fazlul Huq
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Implementation of the BaseSegmenter interface using classical Otsu thresholding.
This wraps the existing `segment_particles` function so it can be used
interchangeably with future AI-based segmenters.
"""

from .base import BaseSegmenter
from .otsu_segment import segment_particles


class OtsuSegmenter(BaseSegmenter):
    def __init__(
        self,
        min_size=3,
        max_size=None,
        save_steps=False,
        output_dir="outputs/segmentation_steps",
        image_name="image",
    ):
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
        self.save_steps = save_steps
        self.output_dir = output_dir
        self.image_name = image_name

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
            binary_image,
            min_size=self.min_size,
            max_size=self.max_size,
            save_steps=self.save_steps,
            output_dir=self.output_dir,
            image_name=self.image_name,
        )
