from typing import List
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from shapes import Shape, SVG, Triangle, Line, Circle
import os
import re
import gzip


class SVGParseException(BaseException):
    pass


def read_svg(fname: str) -> List[Shape]:
    """
    This function is designed to read the test cases and similarly formatted
    files and is not a general purpose SVG parser.
    :param fp: svg file path (string)
    :return: list containing elements of Shape class
    """

    tree = ET.parse(fname)
    root = tree.getroot()
    box = root.attrib["viewBox"].split(" ")
    svg = SVG(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    shapes: List[Shape] = [svg]

    for s in root:
        style = s.attrib["style"].replace(" ", "").split(";")
        color = None
        if "d" in s.attrib:  # triangle
            pts = [x.split(" ") for x in re.split("M|L|Z", s.attrib["d"]) if x]
            pts = np.array([[float(y) for y in x if y] for x in pts])
            for param in style:
                if "fill" in param:
                    rgb = param.split("rgb")[1][1:-1].split(",")
                    color = np.array([float(x) / 255 for x in rgb])
            if color is None:
                raise SVGParseException(f"No fill color found for node:\n{s}")
            shapes.append(Triangle(pts, color))
        elif "r" in s.attrib:  # circle
            center = np.array([float(s.attrib["cx"]), float(s.attrib["cy"])])
            radius = float(s.attrib["r"])
            for param in style:
                if "fill" in param:
                    rgb = param.split("rgb")[1][1:-1].split(",")
                    color = np.array([float(x) / 255 for x in rgb])
            if color is None:
                raise SVGParseException(f"No fill color found for node:\n{s}")
            shapes.append(Circle(center, radius, color))
        else:  # line
            pts = np.array(
                [
                    [float(s.attrib["x1"]), float(s.attrib["y1"])],
                    [float(s.attrib["x2"]), float(s.attrib["y2"])],
                ]
            )
            width = None
            for param in style:
                if "stroke-width" in param:
                    width = float(param.split(":")[1][:-2])
                elif "stroke" in param:
                    rgb = param.split("rgb")[1][1:-1].split(",")
                    color = np.array([float(x) / 255 for x in rgb])
            if color is None:
                raise SVGParseException(f"No fill color found for node:\n{s}")
            if width is None:
                raise SVGParseException(f"No width found for line node:\n{s}")
            shapes.append(Line(pts, width, color))

    return shapes


def img_float_to_uint8(arr: np.ndarray) -> np.ndarray:
    # https://lomont.org/posts/2023/accuratecolorconversions/
    return np.clip(np.floor(arr * 256), 0, 255).astype(np.uint8)


def save_image(fname: str, arr: np.ndarray):
    """
    :param fname: path of where to save the image
    :param arr: numpy array of shape (H,W,3), and should be between 0 and 1

    saves both the image and an .npy.gz file of the original image array
    (we're using gzip compression because our image arrays are big but very compressible)
    """
    im = Image.fromarray(img_float_to_uint8(arr))
    im.save(fname)
    with gzip.GzipFile(os.path.splitext(fname)[0] + ".npy.gz", "w") as f:
        np.save(f, arr)


def load_image_from_npygz(npygz_fname: str) -> np.ndarray:
    with gzip.GzipFile(npygz_fname, "r") as f:
        return np.load(f)


def show_image_diff(a: np.ndarray, b: np.ndarray):
    """
    b is treated as the 'reference' image for tolerance purposes!
    both arrays are (H,W,3) rgb float in range [0,1]
    """
    float_diff = np.any(np.logical_not(np.isclose(a, b, rtol=0, atol=1 / 512)), axis=-1)
    n_different_pixels = np.sum(float_diff)
    print(f"{n_different_pixels} pixels differ between the two images")
    float_diff_image = np.zeros_like(a)
    float_diff_image[:, :, 0] = float_diff.astype(a.dtype)
    # pixels that are different will be in red, same will be black
    cat = np.concatenate((a, b, float_diff_image), axis=1)
    Image.fromarray(img_float_to_uint8(cat)).show()


def show_image_diff_files(a_fname: str, b_fname: str):
    a = (
        load_image_from_npygz(a_fname)
        if a_fname.endswith(".npy.gz")
        else np.asarray(Image.open(a_fname)).astype(np.float64) / 255
    )
    b = (
        load_image_from_npygz(b_fname)
        if b_fname.endswith(".npy.gz")
        else np.asarray(Image.open(b_fname)).astype(np.float64) / 255
    )
    show_image_diff(a, b)


if __name__ == "__main__":
    """
    python utils.py <test image path> <reference image path>

    Each path can be a .png or an .npy.gz array file ((H,W,3) rgb float in range [0,1])
    Shows 3 concatenated images side by side: (test image, reference image, diff image)

    In the diff image, pixels that are different (i.e. outside of float tolerance) between
    test and reference are in red, and pixels that are the same between test and reference
    are in black. You can use this to debug your images.
    """
    import sys

    show_image_diff_files(sys.argv[1], sys.argv[2])
