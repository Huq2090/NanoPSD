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
