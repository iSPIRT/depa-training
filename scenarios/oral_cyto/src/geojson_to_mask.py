# 2025 DEPA Foundation
#
# This work is dedicated to the public domain under the CC0 1.0 Universal license.
# To the extent possible under law, DEPA Foundation has waived all copyright and
# related or neighboring rights to this work.
# CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)
#
# This software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a
# particular purpose and noninfringement. In no event shall the authors or copyright
# holders be liable for any claim, damages or other liability, whether in an action
# of contract, tort or otherwise, arising from, out of or in connection with the
# software or the use or other dealings in the software.
#
# For more information about this framework, please visit:
# https://depa.world/training/depa_training_framework/

# Lifted from https://github.com/<oral_cyto_dataset>/cyto_dataset.py::_geojson_to_mask
# and promoted to a standalone function. No cross-repo import — logic copied per CLAUDE.md.

from typing import Tuple
import numpy as np
import cv2
import geopandas as gpd
from shapely.geometry import mapping


def geojson_to_mask(geojson_path: str, image_shape: Tuple[int, int, int]) -> np.ndarray:
    """Rasterize a GeoJSON of polygons into a uint8 mask matching `image_shape[:2]`.

    Each polygon is filled with 255 (so the PNG is visible and ToTensor() yields {0.0, 1.0}).
    Only the outer ring of the first member of each geometry is used, matching the source.
    """
    gdf = gpd.read_file(geojson_path)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for geom in gdf.geometry:
        if geom is None:
            continue
        coords = np.array(mapping(geom)['coordinates'][0], dtype=np.int32)
        cv2.fillPoly(mask, [coords], 255)
    return mask
