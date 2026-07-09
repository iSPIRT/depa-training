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

# Preprocess TDP B: filter Kaggle oral cytology files by source center, rasterize
# geojson annotations into binary masks, and write per-sample folders matching
# dataset_config.pairing.

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from geojson_to_mask import geojson_to_mask


CENTERS = {"12", "06", "07", "10"}
TDP_NAME = "oral_cyto_B"
TARGET_SIZE = 256

INPUT_ROOT = "/mnt/input/data"
OUTPUT_ROOT = "/mnt/output/preprocessed"


def main():
    input_dir = Path(INPUT_ROOT)
    output_dir = Path(OUTPUT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped_other_center = 0
    skipped_no_pair = 0

    for gj in sorted(input_dir.rglob("*.geojson")):
        stem = gj.stem
        if stem[:2] not in CENTERS:
            skipped_other_center += 1
            continue

        png = gj.with_suffix(".png")
        if not png.exists():
            skipped_no_pair += 1
            continue

        img = np.array(Image.open(png).convert("RGB"))
        mask = geojson_to_mask(str(gj), img.shape)

        img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)

        sample_id = stem
        sample_folder = output_dir / f"oral_cyto_{sample_id}"
        sample_folder.mkdir(parents=True, exist_ok=True)

        Image.fromarray(img_resized).save(sample_folder / f"oral_cyto_{sample_id}_img.png")
        Image.fromarray(mask_resized).save(sample_folder / f"oral_cyto_{sample_id}_seg.png")
        processed += 1

    print(f"Preprocessed {TDP_NAME}: {processed} samples → {OUTPUT_ROOT}")
    print(f"  Skipped (other center): {skipped_other_center}")
    print(f"  Skipped (no png pair):  {skipped_no_pair}")


if __name__ == "__main__":
    main()
