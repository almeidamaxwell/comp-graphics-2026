import raster, os
from pathlib import Path
import numpy as np
from utils import load_image_from_npygz

TEST_DIR = "tests"
OUT_DIR = "output"
TMP_DIR = "tmp"
EXTENSION = ".npy.gz"

outputs = list(Path(OUT_DIR).iterdir())

outputs.sort(key=lambda x: "noaa" in x.name)

try:
    os.mkdir(TMP_DIR)
except OSError as e:
    if e.errno != 17:
        raise e

last_alias = None

for file in outputs:
    if EXTENSION not in file.name:
        continue

    name, ext = file.name.split(EXTENSION)

    test_name, w, h, antialias = name.split("_")

    if antialias != last_alias:
        print("-----")
        print(antialias)
        print("-----")
    last_alias = antialias

    try:
      img_grid = raster.rasterize(
          f"{TEST_DIR}/{test_name}.svg",
          int(w),
          int(h),
          antialias=antialias == "aa"
      )
    except Exception as e:
        print(f"[ERROR] {name} with {e.args}")
        break

    ref_grid = load_image_from_npygz(str(file))

    float_diff = np.any(np.logical_not(np.isclose(img_grid, ref_grid, rtol=0, atol=1 / 512)), axis=-1)
    n_different_pixels = np.sum(float_diff)

    if n_different_pixels > 0:
      print(f"{test_name} ({w}x{h}) {antialias} {n_different_pixels} pixels differ between the two images")

print("Done!")