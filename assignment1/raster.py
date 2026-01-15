from typing import Optional, Tuple, List
import numpy as np
from utils import *
from shapes import Shape, SVG, Triangle, Line, Circle

EPSILON = 1e-15


# NOTE feel free to write your own helper functions as long as they're in raster.py

# returns two functions, one that converts from viewbox to image coordinates and
#   one that convert from image to viewbox coordinates
def create_coords_converters(svg: SVG, im: Tuple[float, float]):
    im_dim = np.array(im)
    svg_dim = np.array((svg.w, svg.h))
    s = im_dim / svg_dim

    # return two functions, convert between viewbox and image coordinates
    def vb_to_im(p: np.ndarray):
        assert isinstance(p, np.ndarray)
        return p * s

    def im_to_vb(p: np.ndarray):
        assert isinstance(p, np.ndarray)
        return p / s

    return vb_to_im, im_to_vb

def clamp(x: float, range: list[float]) -> float:
    return max(range[0], min(range[1], x))


def in_triangle(pt: np.ndarray, tri: np.ndarray):
    s = []

    for i in range(3):
        a = tri[i]
        b = tri[(i + 1) % 3]
        s.append(np.cross(b - a, pt - a))

    return all(x >= -EPSILON for x in s) or all(x <= EPSILON for x in s)


def rasterize(
    svg_file: str,
    im_w: int,
    im_h: int,
    output_file: Optional[str] = None,
    background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    antialias: bool = True,
) -> np.ndarray:
    """
    :param svg_file: filename
    :param im_w: width of image to be rasterized
    :param im_h: height of image to be rasterized
    :param output_file: optional path to save numpy array
    :param background: background color, defaults to white (1, 1 ,1)
    :param antialias: whether to apply antialiasing, defaults to True
    :return: a numpy array of dimension (H,W,3) with RGB values in [0.0,1.0]
    """

    background_arr = np.array(background)
    shapes: List[Shape] = read_svg(svg_file)
    img = np.zeros((im_h, im_w, 3))
    img[:, :, :] = background_arr
    svg = shapes[0]
    assert isinstance(svg, SVG)
    # the first shape in shapes is always the SVG object with the viewbox sizes

    # TODO: put your code here
    vb_to_im, im_to_vb = create_coords_converters(svg, (im_w, im_h))

    queue = shapes[1:]
    while len(queue) > 0:
        shape = queue.pop(0)

        if shape.type == "line":
            assert isinstance(shape, Line)

            a, b = shape.pts

            d = b - a
            norm = np.linalg.norm(d)

            if norm == 0:
                raise Exception("degenerate line")

            d /= norm
            N = np.array([-d[1], d[0]])
            N *= shape.width / 2

            queue.insert(
                0, Triangle(np.array([a - N, a + N, b + N, a - N]), shape.color)
            )
            queue.insert(
                0, Triangle(np.array([b + N, b - N, a - N, b + N]), shape.color)
            )

        elif shape.type == "triangle":
            assert isinstance(shape, Triangle)
            assert np.array_equal(shape.pts[0], shape.pts[-1])

            tri_vb = shape.pts[:-1]
            tri_im = np.array([vb_to_im(v) for v in tri_vb])

            BB_START = (
                int(clamp(np.floor(min(tri_im[:, 0])), [0, im_w - 1])),
                int(clamp(np.floor(min(tri_im[:, 1])), [0, im_h - 1])),
            )
            BB_END = (
                int(clamp(np.ceil(max(tri_im[:, 0])), [0, im_w - 1])),
                int(clamp(np.ceil(max(tri_im[:, 1])), [0, im_h - 1])),
            )

            for j in range(BB_START[1], BB_END[1]):
                for i in range(BB_START[0], BB_END[0]):
                    # not antialiasing case
                    if not antialias and in_triangle(
                        im_to_vb(np.array([i + 0.5, j + 0.5])), tri_vb
                    ):
                        img[j, i, :] = shape.color
                        continue

                    # antialiasing case
                    a = 0
                    K = 3
                    for t in np.linspace(j, j + 1, K):
                        for u in np.linspace(i, i + 1, K):
                            if in_triangle(im_to_vb(np.array([u, t])), tri_vb):
                                a += 1

                    a /= K ** 2
                    img[j, i] = (1 - a) * img[j, i] + a * shape.color

        elif shape.type == "circle":
            assert isinstance(shape, Circle)

            r_vb = shape.radius
            r_im = vb_to_im(
                np.array([shape.radius, shape.radius])
            )  # two radii because image could be non-square
            c_vb = shape.center
            c_im = vb_to_im(c_vb)

            BB_START = (
                int(clamp(np.floor(c_im[0] - r_im[0]), [0, im_w - 1])),
                int(clamp(np.floor(c_im[1] - r_im[1]), [0, im_h - 1])),
            )
            BB_END = (
                int(clamp(np.ceil(c_im[0] + r_im[0]), [0, im_w - 1])),
                int(clamp(np.ceil(c_im[1] + r_im[1]), [0, im_h - 1])),
            )

            for j in range(BB_START[1], BB_END[1]):
                for i in range(BB_START[0], BB_END[0]):
                    if not antialias:
                        x, y = im_to_vb(np.array([i + 0.5, j + 0.5]))
                        if (x - c_vb[0]) ** 2 + (y - c_vb[1]) ** 2 > r_vb**2:
                            continue

                        img[j, i] = shape.color
                        continue

                    a = 0
                    K = 3
                    for t in np.linspace(j, j + 1, K):
                        for u in np.linspace(i, i + 1, K):
                            x, y = im_to_vb(np.array([u, t]))
                            if (x - c_vb[0]) ** 2 + (y - c_vb[1]) ** 2 <= r_vb**2:
                              a += 1

                    a /= K ** 2
                    img[j, i] = (1 - a) * img[j, i] + a * shape.color

    if output_file:
        save_image(output_file, img)

    return img


if __name__ == "__main__":
    rasterize(
        "tests/test6.svg", 128, 128, output_file="your_output.png", antialias=False
    )
