from typing import Sequence, Optional
import os
import numpy as np
from PIL import Image
import gzip


class TriangleMesh:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_colors: Optional[np.ndarray] = None,
        vertex_colors: Optional[np.ndarray] = None,
    ):
        """
        vertices: (n_vertices, 3) float array of vertex positions
        faces: (n_faces, 3) int array of indices into vertices, each row the verts of a triangle
        face_colors: (n_faces, 3) float array of a rgb color (in range [0,1]) per face
        vertex_colors: (n_vertices, 3) float array of a rgb color (in range [0,1]) per vertex
        """
        self.vertices = vertices
        self.faces = faces
        self.face_colors = face_colors
        self.vertex_colors = vertex_colors


def save_image(fname: str, arr: np.ndarray) -> np.ndarray:
    """
    :param fname: path of where to save the image
    :param arr: numpy array of shape (H,W,3), and should be between 0 and 1

    saves both the image and an .npy.gz file of the original image array
    and returns back the original array
    """
    im = Image.fromarray(np.clip(np.floor(arr * 256), 0, 255).astype(np.uint8))
    im.save(fname)
    with gzip.GzipFile(os.path.splitext(fname)[0] + ".npy.gz", "w") as f:
        np.save(f, arr)
    return arr


def read_image(fname: str) -> np.ndarray:
    """reads image file and returns as numpy array (H,W,3) rgb in range [0,1]"""
    return np.asarray(Image.open(fname)).astype(np.float64) / 255


"""
The following functions (make_*, calc_*, and update_zbuffer) are merely a
suggested outline for planning out your solution. They should give you an idea
of the subtasks involved and how you might factor out common code between parts.
You are free to modify these functions and organize your code however you like.
"""


def make_viewport_matrix(im_h: int, im_w: int) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    pass


def make_orthographic_matrix(
    l: float = 0.0,
    r: float = 12.0,
    b: float = 0.0,
    t: float = 12.0,
    n: float = 12.0,
    f: float = 0.0,
) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.

    (These default argument values are the orthographic view volume parameters
    for P2 and P3.)
    """
    pass


def make_camera_matrix(eye: np.ndarray, lookat: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    pass


def make_perspective_matrix(fovy: float, aspect: float, n: float, f: float) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    pass


def calc_coverage(face_in_image_space, test_pixel_x, test_pixel_y):
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    pass


def calc_triangle_bounding_box(face: np.ndarray):
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    pass


def update_zbuffer(zbuffer: np.ndarray, YOUR_OTHER_ARGUMENTS_ETC):
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    pass


"""
The functions below are the ones actually run for grading. Do not change the
signatures of these functions. The autograder will run them and expect the
result image to be returned from them.
"""


# P1
def render_viewport(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """
    Render out just the vertices of each triangle in the input object.
    TIP: Pad the vertex pixel out in order to visualize properly like in the
    handout pdf (but turn that off when you submit your code)
    """
    return save_image("p1.png", YOUR_IMAGE_ARRAY_HERE)


# P2
def render_ortho(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the orthographic projection of the cube"""
    return save_image("p2.png", YOUR_IMAGE_ARRAY_HERE)


# P3
def render_camera(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the orthographic projection of the cube with the specific camera settings"""
    return save_image("p3.png", YOUR_IMAGE_ARRAY_HERE)


# P4
def render_perspective(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the perspective projection with perspective divide"""
    return save_image("p4.png", YOUR_IMAGE_ARRAY_HERE)


# P5
def render_zbuffer_with_color(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the input with z-buffering and color interpolation enabled"""
    return save_image("p5.png", YOUR_IMAGE_ARRAY_HERE)


# P6
def render_big_scene(objlist: Sequence[TriangleMesh], im_w: int, im_h: int) -> np.ndarray:
    """Render a big scene with multiple shapes"""
    return save_image("p6.png", YOUR_IMAGE_ARRAY_HERE)


# P7
def my_cube_uvs(cube: TriangleMesh) -> np.ndarray:
    """
    Build your own UV coordinates for the cube mesh to test out texture_map.
    You may choose to hard-code numbers or compute a planar parameterization of
    the cube mesh. The UVs should have shape (n_faces, 3, 2), i.e. UV coordinates
    (u,v) for each of the 3 corners of each face.

    Note that this function is for you to use to make input UVs for running your
    implementation of texture_map, for reproducing p7.png, and for creating your
    custom.png. The autograder will not be running this. The autograder will
    call texture_map with our UVs.
    """
    # You may find it helpful to start with this array and fill it out with correct values.
    uvs = np.zeros((len(cube.faces), 3, 2))
    return uvs


def texture_map(
    obj: TriangleMesh, uvs: np.ndarray, img: np.ndarray, im_w: int, im_h: int
) -> np.ndarray:
    """
    Render a cube with the texture img mapped onto its faces according to uvs.
    `uvs` has shape (n_faces, 3, 2) and contains the UV coordinates (u,v) for
    each of the 3 corners of each face
    `img` has shape (height, width, 3)
    """
    return save_image("p7.png", YOUR_IMAGE_ARRAY_HERE)


####### setup stuff
def get_big_scene():
    # Cube
    vertices = np.array(
        [
            [-0.35, -0.35, -0.15],
            [-0.15, -0.35, -0.15],
            [-0.35, -0.15, -0.15],
            [-0.15, -0.15, -0.15],
            [-0.35, -0.35, -0.35],
            [-0.15, -0.35, -0.35],
            [-0.35, -0.15, -0.35],
            [-0.15, -0.15, -0.35],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [5, 4, 7],
            [7, 4, 6],
            [4, 0, 6],
            [6, 0, 2],
            [1, 5, 3],
            [3, 5, 7],
            [2, 3, 6],
            [6, 3, 7],
            [4, 5, 0],
            [0, 5, 1],
        ]
    )
    vertex_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    cube1 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    # Tet
    vertices = np.array(
        [[0.45, 0.5, 0.35], [0.4, 0.4, 0.45], [0.4, 0.35, 0.25], [0.4, 0.45, 0.3]]
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    vertex_colors = np.array(
        [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
    )
    tet1 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    # Tet
    vertices = np.array(
        [[0.0, 0.0, 0.0], [-0.1, -0.3, -0.25], [-0.1, 0.1, 0.3], [-0.1, -0.15, 0.4]]
    )
    triangles = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    vertex_colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]
    )
    tet2 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    vertices = np.array(
        [
            [-0.4, -0.4, 0.2],
            [-0.5, -0.4, 0.2],
            [-0.4, -0.5, 0.2],
            [-0.5, -0.5, 0.2],
            [-0.4, -0.4, 0.3],
            [-0.5, -0.4, 0.3],
            [-0.4, -0.5, 0.3],
            [-0.5, -0.5, 0.3],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [5, 4, 7],
            [7, 4, 6],
            [4, 0, 6],
            [6, 0, 2],
            [1, 5, 3],
            [3, 5, 7],
            [2, 3, 6],
            [6, 3, 7],
            [4, 5, 0],
            [0, 5, 1],
        ]
    )
    vertex_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )

    cube2 = TriangleMesh(vertices, triangles, vertex_colors=vertex_colors)

    return [cube1, tet1, tet2, cube2]


if __name__ == "__main__":
    im_w = 800
    im_h = 600
    vertices = np.array(
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [5, 4, 7],
            [7, 4, 6],
            [4, 0, 6],
            [6, 0, 2],
            [1, 5, 3],
            [3, 5, 7],
            [2, 3, 6],
            [6, 3, 7],
            [4, 5, 0],
            [0, 5, 1],
        ]
    )
    triangle_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    cube = TriangleMesh(vertices, triangles, triangle_colors)

    # NOTE for your own testing purposes:
    # Uncomment and run each of these commented-out functions after you've filled them out

    # render_viewport(cube, im_w, im_h)
    ortho_vertices = np.array(
        [
            [1.0, 1.0, 1.5],
            [11.0, 1.0, 1.5],
            [1.0, 11.0, 1.5],
            [11.0, 11.0, 1.5],
            [1.0, 1.0, -1.5],
            [11.0, 1.0, -1.5],
            [1.0, 11.0, -1.5],
            [11.0, 11.0, -1.5],
        ]
    )
    ortho_cube = TriangleMesh(ortho_vertices, triangles, triangle_colors)
    # render_ortho(ortho_cube, im_w, im_h)
    # render_camera(ortho_cube, im_w, im_h)
    # render_perspective(cube, im_w, im_h)
    vertex_colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    cube.vertex_colors = vertex_colors
    # render_zbuffer_with_color(cube, im_w, im_h)

    objlist = get_big_scene()
    # render_big_scene(objlist, im_w, im_h)
    img = read_image("flag.png")
    uvs = my_cube_uvs(cube)
    # texture_map(cube, uvs, img, im_w, im_h)
