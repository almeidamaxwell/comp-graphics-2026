from typing import Optional, Tuple, List
import numpy as np
from utils import *
from shapes import Shape, SVG, Triangle, Line, Circle

# NOTE feel free to write your own helper functions as long as they're in raster.py

K = 3


# returns two functions, one that converts from viewbox to image coordinates and
#   one that convert from image to viewbox coordinates
def im_svg_change_of_basis(svg: SVG, im: Tuple[float, float]):
    im_dim = np.array(im)
    svg_dim = np.array((svg.w, svg.h))
    s = im_dim / svg_dim

    return s, 1 / s


def in_triangle(P: np.ndarray, tri: np.ndarray, eps=1e-12):
    assert isinstance(P, np.ndarray)
    assert isinstance(tri, np.ndarray)

    A, B, C = tri

    def edge(U, V, X):
        return (V[0] - U[0]) * (X[..., 1] - U[1]) - (V[1] - U[1]) * (X[..., 0] - U[0])

    e0 = edge(A, B, P)
    e1 = edge(B, C, P)
    e2 = edge(C, A, P)

    return ((e0 >= -eps) & (e1 >= -eps) & (e2 >= -eps)) | (
        (e0 <= eps) & (e1 <= eps) & (e2 <= eps)
    )


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
    vb_to_im, im_to_vb = im_svg_change_of_basis(svg, (im_w, im_h))

    def triangle_inside_mask(tri_im, bounds):
        x_start, x_end, y_start, y_end = bounds

        X, Y = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))
        offs = np.linspace(0.0, 1.0, K, endpoint=True)

        U = X[..., None, None] + offs[None, None, None, :]
        T = Y[..., None, None] + offs[None, None, :, None]

        U, T = np.broadcast_arrays(U, T)
        P = np.stack([U, T], axis=-1)
        return in_triangle(P, tri_im)

    def compute_triangle_coverage(
        tri_im: np.ndarray, tri_vb: np.ndarray, bounds: list[float]
    ) -> np.ndarray | None:
        x_start, x_end, y_start, y_end = bounds

        if x_end <= x_start or y_end <= y_start:
            return None

        X, Y = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))

        if not antialias:
            P_im = np.stack([X + 0.5, Y + 0.5], axis=-1)
            P_vb = P_im * im_to_vb

            inside = in_triangle(P_vb, tri_vb)
            return inside.astype(np.float64)
        else:
            offs = np.linspace(0.0, 1.0, K, endpoint=True)

            U = X[..., None, None] + offs[None, None, None, :]
            T = Y[..., None, None] + offs[None, None, :, None]

            U, T = np.broadcast_arrays(U, T)
            P_im = np.stack([U, T], axis=-1)
            P_vb = P_im * im_to_vb

            inside = in_triangle(P_vb, tri_vb)
            return inside.mean(axis=(-1, -2))

    for shape in shapes[1:]:
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

            tris_vb = np.array(
                [
                    np.array([a - N, a + N, b + N]),
                    np.array([b + N, b - N, a - N]),
                ]
            )
            tris_im = tris_vb * vb_to_im

            # handle no antialiasing case

            pts = tris_im.reshape(-1, 2)

            x_start = int(np.clip(np.floor(pts[:, 0].min()), 0, im_w - 1))
            x_end = int(np.clip(np.ceil(pts[:, 0].max()), 0, im_w))
            y_start = int(np.clip(np.floor(pts[:, 1].min()), 0, im_h - 1))
            y_end = int(np.clip(np.ceil(pts[:, 1].max()), 0, im_h))

            inside_union = None
            for tri_im, tri_vb in zip(tris_im, tris_vb):
                inside_i = triangle_inside_mask(
                    tri_im, [x_start, x_end, y_start, y_end]
                )
                inside_union = (
                    inside_i if inside_union is None else (inside_union | inside_i)
                )

            if inside_union is None:
                continue

            a = inside_union.mean(axis=(-1, -2))

            box = img[y_start:y_end, x_start:x_end, :]
            img[y_start:y_end, x_start:x_end, :] = (1.0 - a)[..., None] * box + a[
                ..., None
            ] * shape.color[None, None, :]

        elif shape.type == "triangle":
            assert isinstance(shape, Triangle)
            assert np.array_equal(shape.pts[0], shape.pts[-1])

            tri_vb = shape.pts[:-1]
            tri_im = tri_vb * vb_to_im

            x_start = int(np.clip(np.floor(tri_im[:, 0].min()), 0, im_w - 1))
            x_end = int(np.clip(np.ceil(tri_im[:, 0].max()), 0, im_w))
            y_start = int(np.clip(np.floor(tri_im[:, 1].min()), 0, im_h - 1))
            y_end = int(np.clip(np.ceil(tri_im[:, 1].max()), 0, im_h))

            a = compute_triangle_coverage(
                tri_im, tri_vb, [x_start, x_end, y_start, y_end]
            )

            if a is None:
                continue

            box = img[y_start:y_end, x_start:x_end, :]
            img[y_start:y_end, x_start:x_end, :] = (1.0 - a)[..., None] * box + a[
                ..., None
            ] * shape.color[None, None, :]

        elif shape.type == "circle":
            assert isinstance(shape, Circle)

            r_vb = shape.radius
            # two radii because image could be non-square
            r_im = np.array([shape.radius, shape.radius]) * vb_to_im
            c_vb = shape.center
            c_im = c_vb * vb_to_im

            x_start = int(np.clip(np.floor(c_im[0] - r_im[0]), 0, im_w - 1))
            x_end = int(np.clip(np.ceil(c_im[0] + r_im[0]), 0, im_w))
            y_start = int(np.clip(np.floor(c_im[1] - r_im[1]), 0, im_h - 1))
            y_end = int(np.clip(np.ceil(c_im[1] + r_im[1]), 0, im_h))

            if x_end <= x_start or y_end <= y_start:
                continue  # need reason

            X, Y = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))

            if not antialias:
                P_im = np.stack([X + 0.5, Y + 0.5], axis=-1)
                P_vb = P_im * im_to_vb
                dx = P_vb[..., 0] - c_vb[0]
                dy = P_vb[..., 1] - c_vb[1]
                inside = dx * dx + dy * dy <= r_vb * r_vb
                a = inside.astype(np.float64)
            else:
                offs = np.linspace(0.0, 1.0, K, endpoint=True)
                U = X[..., None, None] + offs[None, None, None, :]
                T = Y[..., None, None] + offs[None, None, :, None]

                U, T = np.broadcast_arrays(U, T)

                P_im = np.stack([U, T], axis=-1)
                P_vb = P_im * im_to_vb

                dx = P_vb[..., 0] - c_vb[0]
                dy = P_vb[..., 1] - c_vb[1]
                inside = dx * dx + dy * dy <= r_vb * r_vb
                a = inside.mean(axis=(-1, -2))

            box = img[y_start:y_end, x_start:x_end, :]
            img[y_start:y_end, x_start:x_end, :] = (1.0 - a)[..., None] * box + a[
                ..., None
            ] * shape.color[None, None, :]

    if output_file:
        save_image(output_file, img)

    return img


if __name__ == "__main__":
    rasterize(
        "tests/test6.svg", 128, 128, output_file="your_output.png", antialias=False
    )
