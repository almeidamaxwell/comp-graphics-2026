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
    n_x = float(im_w)
    n_y = float(im_h)
    return np.array(
        [
            [n_x / 2.0, 0.0, 0.0, n_x / 2.0],
            [0.0, -n_y / 2.0, 0.0, n_y / 2.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


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
    return np.array(
        [
            [2.0 / (r - l), 0.0, 0.0, -(r + l) / (r - l)],
            [0.0, 2.0 / (t - b), 0.0, -(t + b) / (t - b)],
            [0.0, 0.0, 2.0 / (n - f), -(n + f) / (n - f)],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def make_camera_matrix(
    eye: np.ndarray, lookat: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    w = eye - lookat
    w /= np.linalg.norm(w)

    u = np.cross(up, w)
    u /= np.linalg.norm(u)

    v = np.cross(w, u)

    return np.array(
        [
            [*u, -np.dot(u, eye)],
            [*v, -np.dot(v, eye)],
            [*w, -np.dot(w, eye)],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def make_perspective_matrix(
    fovy: float, aspect: float, n: float, f: float
) -> np.ndarray:
    """
    You're free to modify this function's signature; this is merely a suggested
    way to factor out a common subtask.
    """
    theta = np.deg2rad(fovy)
    t = abs(n) * np.tan(theta / 2.0)
    r = t * aspect

    return np.array(
        [
            [-n / r, 0.0, 0.0, 0.0],
            [0.0, -n / t, 0.0, 0.0],
            [0.0, 0.0, (n + f) / (n - f), 2 * n * f / (n - f)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )


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


def barycentric(P: np.ndarray, tri: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    assert isinstance(P, np.ndarray)
    assert isinstance(tri, np.ndarray)

    A, B, C = tri

    def edge(U, V, X):
        return (V[0] - U[0]) * (X[..., 1] - U[1]) - (V[1] - U[1]) * (X[..., 0] - U[0])

    area = edge(A, B, C)
    if abs(area) < eps:  # degenerate triangle
        return np.array([0.0, 0.0, 0.0])

    return np.array([edge(B, C, P) / area, edge(C, A, P) / area, edge(A, B, P) / area])


def in_triangle(P: np.ndarray, tri: np.ndarray, eps=1e-12):
    alpha, beta, gamma = barycentric(P, tri, eps)
    return ((alpha >= -eps) & (beta >= -eps) & (gamma >= -eps)) | (
        (alpha <= eps) & (beta <= eps) & (gamma <= eps)
    )


def floorclip(val: float, min: int, max: int):
    """
    Floors the value and then clips it
    """
    return int(np.clip(np.floor(val), min, max))


def ceilclip(val: float, min: int, max: int):
    """
    Ceils the value and then clips it
    """
    return int(np.clip(np.ceil(val), min, max))


def roundclip(val: float, min: int, max: int):
    """
    Rounds the value and then clips it
    """
    return int(np.clip(np.round(val), min, max))


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
    assert obj.face_colors is not None

    img = np.zeros((im_h, im_w, 3))

    M_vp = make_viewport_matrix(im_h, im_w)

    for l, F_l in enumerate(obj.faces):
        color = obj.face_colors[l]

        for k in range(3):
            p = obj.vertices[F_l[k]]
            p_H = np.array([*p, 1.0])
            s_H = M_vp @ p_H
            x_scr, y_scr, *_ = s_H

            i = int(y_scr)
            j = int(x_scr)

            if 0 <= i < im_h and 0 <= j < im_w:
                img[i, j, :] = color

                # pad = 3
                # i0, i1 = max(0, i - pad), min(im_h, i + pad + 1)
                # j0, j1 = max(0, j - pad), min(im_w, j + pad + 1)
                # img[i0:i1, j0:j1, :] = color

    return save_image("p1.png", img)


# P2
def render_ortho(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the orthographic projection of the cube"""
    assert obj.face_colors is not None

    img = np.zeros((im_h, im_w, 3))

    M_ortho = make_orthographic_matrix()
    M_vp = make_viewport_matrix(im_h, im_w)

    for l, F_l in enumerate(obj.faces):
        color = obj.face_colors[l]

        # transform vertices into screen space
        pts = []
        for k in range(3):
            p = obj.vertices[F_l[k]]
            p_H = np.array([*p, 1.0])
            p_can = M_ortho @ p_H
            p_scr = M_vp @ p_can
            pts.append(p_scr[:2])

        pts = np.array(pts)

        xmin = floorclip(np.min(pts[:, 0]), 0, im_w - 1)
        xmax = ceilclip(np.max(pts[:, 0]), 0, im_w - 1)
        ymin = floorclip(np.min(pts[:, 1]), 0, im_h - 1)
        ymax = ceilclip(np.max(pts[:, 1]), 0, im_h - 1)

        for i in range(ymin, ymax + 1):
            for j in range(xmin, xmax + 1):
                if in_triangle(np.array([j + 0.5, i + 0.5]), pts):
                    img[i, j, :] = color

    return save_image("p2.png", img)


# P3
def render_camera(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the orthographic projection of the cube with the specific camera settings"""
    assert obj.face_colors is not None

    img = np.zeros((im_h, im_w, 3))

    M_cam = make_camera_matrix(
        eye=np.array([0.2, 0.2, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    M_ortho = make_orthographic_matrix()
    M_vp = make_viewport_matrix(im_h, im_w)

    for l, F_l in enumerate(obj.faces):
        color = obj.face_colors[l]

        # transform vertices into screen space
        pts = []
        for k in range(3):
            p = obj.vertices[F_l[k]]
            p_H = np.array([*p, 1.0])
            p_cam = M_cam @ p_H
            p_can = M_ortho @ p_cam
            p_scr = M_vp @ p_can
            pts.append(p_scr[:2])

        pts = np.array(pts)

        xmin = floorclip(np.min(pts[:, 0]), 0, im_w - 1)
        xmax = ceilclip(np.max(pts[:, 0]), 0, im_w - 1)
        ymin = floorclip(np.min(pts[:, 1]), 0, im_h - 1)
        ymax = ceilclip(np.max(pts[:, 1]), 0, im_h - 1)

        for i in range(ymin, ymax + 1):
            for j in range(xmin, xmax + 1):
                if in_triangle(np.array([j + 0.5, i + 0.5]), pts):
                    img[i, j, :] = color

    return save_image("p3.png", img)


# P4
def render_perspective(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the perspective projection with perspective divide"""
    assert obj.face_colors is not None

    img = np.zeros((im_h, im_w, 3))

    M_cam = make_camera_matrix(
        eye=np.array([1.0, 1.0, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    M_per = make_perspective_matrix(65.0, 4 / 3, -1.0, -100.0)
    M_vp = make_viewport_matrix(im_h, im_w)

    for l, F_l in enumerate(obj.faces):
        color = obj.face_colors[l]

        # transform vertices into screen space
        pts = []
        for k in range(3):
            p = obj.vertices[F_l[k]]
            p_H = np.array([*p, 1.0])
            p_cam = M_cam @ p_H
            p_clip = M_per @ p_cam

            # perspective divide
            p_ndc = p_clip[:3] / p_clip[3]
            p_ndc_H = np.array([*p_ndc, 1.0])

            p_scr = M_vp @ p_ndc_H
            pts.append(p_scr[:2])

        pts = np.array(pts)

        xmin = floorclip(np.min(pts[:, 0]), 0, im_w - 1)
        xmax = ceilclip(np.max(pts[:, 0]), 0, im_w - 1)
        ymin = floorclip(np.min(pts[:, 1]), 0, im_h - 1)
        ymax = ceilclip(np.max(pts[:, 1]), 0, im_h - 1)

        for i in range(ymin, ymax + 1):
            for j in range(xmin, xmax + 1):
                if in_triangle(np.array([j + 0.5, i + 0.5]), pts):
                    img[i, j, :] = color

    return save_image("p4.png", img)


# P5
def render_zbuffer_with_color(obj: TriangleMesh, im_w: int, im_h: int) -> np.ndarray:
    """Render the input with z-buffering and color interpolation enabled"""
    assert obj.vertex_colors is not None

    img = np.zeros((im_h, im_w, 3))
    zbuf = np.full((im_h, im_w), -np.inf)

    M_cam = make_camera_matrix(
        eye=np.array([1.0, 1.0, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    M_per = make_perspective_matrix(65.0, 4 / 3, -1.0, -100.0)
    M_vp = make_viewport_matrix(im_h, im_w)

    for l, F_l in enumerate(obj.faces):
        pts = []
        zs = []
        cols = []

        for m in F_l:
            p = obj.vertices[m]
            c = obj.vertex_colors[m]

            p_H = np.array([*p, 1.0])
            p_cam = M_cam @ p_H
            p_clip = M_per @ p_cam

            # perspective divide
            p_ndc = p_clip[:3] / p_clip[3]
            p_ndc_H = np.array([*p_ndc, 1.0])

            zs.append(p_ndc[2])
            cols.append(c)

            p_scr = M_vp @ p_ndc_H
            pts.append(p_scr[:2])

        pts = np.array(pts)
        zs = np.array(zs)
        cols = np.array(cols)

        xmin = floorclip(np.min(pts[:, 0]), 0, im_w - 1)
        xmax = ceilclip(np.max(pts[:, 0]), 0, im_w - 1)
        ymin = floorclip(np.min(pts[:, 1]), 0, im_h - 1)
        ymax = ceilclip(np.max(pts[:, 1]), 0, im_h - 1)

        for i in range(ymin, ymax + 1):
            for j in range(xmin, xmax + 1):
                P = np.array([j + 0.5, i + 0.5])
                if in_triangle(P, pts):
                    P_bary = barycentric(P, pts)
                    z = P_bary.dot(zs)
                    if z > zbuf[i, j]:
                        zbuf[i, j] = z
                        img[i, j, :] = P_bary.dot(cols)

    return save_image("p5.png", img)


# P6
def render_big_scene(
    objlist: Sequence[TriangleMesh], im_w: int, im_h: int
) -> np.ndarray:
    """Render a big scene with multiple shapes"""
    img = np.zeros((im_h, im_w, 3))
    zbuf = np.full((im_h, im_w), -np.inf)

    M_cam = make_camera_matrix(
        eye=np.array([-0.5, 1.0, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    M_per = make_perspective_matrix(65.0, 4 / 3, -1.0, -100.0)
    M_vp = make_viewport_matrix(im_h, im_w)

    for obj in objlist:
        assert obj.vertex_colors is not None

        for l, F_l in enumerate(obj.faces):
            pts = []
            zs = []
            cols = []

            for m in F_l:
                p = obj.vertices[m]
                c = obj.vertex_colors[m]

                p_H = np.array([*p, 1.0])
                p_cam = M_cam @ p_H
                p_clip = M_per @ p_cam

                # perspective divide
                p_ndc = p_clip[:3] / p_clip[3]
                p_ndc_H = np.array([*p_ndc, 1.0])

                zs.append(p_ndc[2])
                cols.append(c)

                p_scr = M_vp @ p_ndc_H
                pts.append(p_scr[:2])

            pts = np.array(pts)
            zs = np.array(zs)
            cols = np.array(cols)

            xmin = floorclip(np.min(pts[:, 0]), 0, im_w - 1)
            xmax = ceilclip(np.max(pts[:, 0]), 0, im_w - 1)
            ymin = floorclip(np.min(pts[:, 1]), 0, im_h - 1)
            ymax = ceilclip(np.max(pts[:, 1]), 0, im_h - 1)

            for i in range(ymin, ymax + 1):
                for j in range(xmin, xmax + 1):
                    P = np.array([j + 0.5, i + 0.5])
                    if in_triangle(P, pts):
                        P_bary = barycentric(P, pts)
                        z = P_bary.dot(zs)
                        if z > zbuf[i, j]:
                            zbuf[i, j] = z
                            img[i, j, :] = P_bary.dot(cols)

    return save_image("p6.png", img)


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

    for f in range(0, len(cube.faces), 2):
        uvs[f + 0] = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        uvs[f + 1] = np.array([
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])

    return uvs


def texture_map(
    obj: TriangleMesh, uvs: np.ndarray, tex: np.ndarray, im_w: int, im_h: int
) -> np.ndarray:
    """
    Render a cube with the texture img mapped onto its faces according to uvs.
    `uvs` has shape (n_faces, 3, 2) and contains the UV coordinates (u,v) for
    each of the 3 corners of each face
    `img` has shape (height, width, 3)
    """
    tex_h, tex_w = tex.shape[:2]
    img = np.zeros((im_h, im_w, 3))
    zbuf = np.full((im_h, im_w), -np.inf)

    M_cam = make_camera_matrix(
        eye=np.array([1.0, 1.0, 1.0]),
        lookat=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    M_per = make_perspective_matrix(65.0, 4 / 3, -1.0, -100.0)
    M_vp = make_viewport_matrix(im_h, im_w)

    for l, F_l in enumerate(obj.faces):
        pts = []
        zs = []
        u_pw = []
        v_pw = []
        q = []

        for k in range(3):
            p = obj.vertices[F_l[k]]
            u, v = uvs[l, k]

            p_H = np.array([*p, 1.0])
            p_cam = M_cam @ p_H
            p_clip = M_per @ p_cam

            w = p_clip[3]
            u_pw.append(u / w)
            v_pw.append(v / w)
            q.append(1.0 / w)

            # perspective divide
            p_ndc = p_clip[:3] / w
            p_ndc_H = np.array([*p_ndc, 1.0])

            zs.append(p_ndc[2])

            p_scr = M_vp @ p_ndc_H
            pts.append(p_scr[:2])

        pts = np.array(pts)
        zs = np.array(zs)
        u_pw = np.array(u_pw)
        v_pw = np.array(v_pw)
        q = np.array(q)

        xmin = floorclip(np.min(pts[:, 0]), 0, im_w - 1)
        xmax = ceilclip(np.max(pts[:, 0]), 0, im_w - 1)
        ymin = floorclip(np.min(pts[:, 1]), 0, im_h - 1)
        ymax = ceilclip(np.max(pts[:, 1]), 0, im_h - 1)

        for i in range(ymin, ymax + 1):
            for j in range(xmin, xmax + 1):
                P = np.array([j + 0.5, i + 0.5])
                if not in_triangle(P, pts):
                    continue

                P_bary = barycentric(P, pts)
                z = P_bary.dot(zs)

                if z <= zbuf[i, j]:
                    continue

                zbuf[i, j] = z

                q_interp = P_bary.dot(q)
                u = P_bary.dot(u_pw) / q_interp
                v = P_bary.dot(v_pw) / q_interp

                t_i = roundclip((1 - v) * tex_h - 0.5, 0, tex_h - 1)
                t_j = roundclip(u * tex_w - 0.5, 0, tex_w - 1)

                img[i, j] = tex[t_i, t_j]

    return save_image("p7.png", img)


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
    texture_map(cube, uvs, img, im_w, im_h)
