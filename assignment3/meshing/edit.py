from typing import Optional
from dataclasses import dataclass, field
from . import Halfedge, Edge, Vertex, Face, Topology, Mesh
import numpy as np


"""
TODO complete these functions
P5 -- LaplacianSmoothing.apply
P6 -- prepare_collapse, do_collapse
Extra credit -- link_condition
"""


class MeshEdit:
    """
    Abstract interface for a mesh edit. The edit is prepared upon init
    (creating/storing info about the edit before actually executing it) then, if
    determined to be doable, applied with apply().
    """

    def __init__(self):
        pass

    def apply(self):
        pass


class LaplacianSmoothing(MeshEdit):
    def __init__(self, mesh: Mesh, n_iter: int):
        self.mesh = mesh
        self.n_iter = n_iter

    def apply(self):
        # TODO: P5 -- complete this function
        for _ in range(self.n_iter):
            new_vertices = self.mesh.vertices.copy()

            for vertex in self.mesh.topology.vertices.values():
                # compute position as average of neighbors
                neighbors = list(map(self.mesh.get_3d_pos, vertex.adjacentVertices()))
                new_vertices[vertex.index] = np.mean(neighbors, axis=0)

            self.mesh.vertices = new_vertices


class EdgeCollapse(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id
        self.prep = prepare_collapse(self.mesh, self.e_id)

    def apply(self):
        return do_collapse(self.prep, self.mesh)


@dataclass
class CollapsePrep:
    """
    A data-class that stores all the operations you may need to perform during
    an edge collapse.

    The intention of this data-class is to help keep everything organized and
    remind you of what aspects of the half-edge mesh structure you need to keep
    track of. Depending on your implementation, you very likely will not need to
    use all of the parameters below.

    Read this link to learn more about data-classes in Python:
    https://www.dataquest.io/blog/how-to-use-python-data-classes/
    """

    # The vertices that need to be merged through the edge collapse
    merge_verts: tuple[Vertex, Vertex]

    # The primitives that need their references updated. Each list item is a
    # tuple (primitive_that_needs_a_reference_fix, new_primitive_it_should_point_to)
    #
    # The field(default_factory=list) default value just means that each field will be
    # initialized as an empty list which is instantiated when this dataclass is instantiated
    repair_he_twins: list[tuple[Halfedge, Halfedge]] = field(default_factory=list)
    repair_he_nexts: list[tuple[Halfedge, Halfedge]] = field(default_factory=list)
    repair_he_edges: list[tuple[Halfedge, Edge]] = field(default_factory=list)
    repair_he_verts: list[tuple[Halfedge, Vertex]] = field(default_factory=list)
    repair_he_faces: list[tuple[Halfedge, Face]] = field(default_factory=list)
    repair_edge_hes: list[tuple[Edge, Halfedge]] = field(default_factory=list)
    repair_vert_hes: list[tuple[Vertex, Halfedge]] = field(default_factory=list)
    repair_face_hes: list[tuple[Face, Halfedge]] = field(default_factory=list)

    # The primitives that need to be deleted
    del_verts: list[Vertex] = field(default_factory=list)
    del_edges: list[Edge] = field(default_factory=list)
    del_hes: list[Halfedge] = field(default_factory=list)
    del_faces: list[Face] = field(default_factory=list)


# TODO: P6 -- complete this
def prepare_collapse(mesh: Mesh, e_id: int) -> CollapsePrep:
    """
    The first stage of edge-collapse.

    This function should traverse the mesh's topology and figure out which
    operations are needed to perform an edge collapse. These operations should
    be stored in a `CollapsePrep` object (see definition above) that is returned.
    """
    topology = mesh.topology
    e = topology.edges[e_id]
    # TODO write your code here, replace this raise, and return a `CollapsePrep`

    cd = e.halfedge
    dc = cd.twin

    db = cd.next
    bd = db.twin

    ca = dc.next
    ac = ca.twin

    ad = dc.prev()
    da = ad.twin

    bc = cd.prev()
    cb = bc.twin

    a = ad.vertex
    b = bd.vertex
    c = cd.vertex
    d = dc.vertex

    prep = CollapsePrep((c, d))

    # delete half edges of the triangles
    prep.del_hes += [cd, db, bc, dc, ca, ad]
    # delete faces of the triangles
    prep.del_faces.extend([cd.face, dc.face])
    # delete edges that connect to d
    prep.del_edges.extend([e, ad.edge, db.edge])
    # delete vertex d
    prep.del_verts.append(d)

    # repair neighbors of d
    he = bd.next
    while True:
        prep.repair_he_verts.append((he, c))
        he = he.twin.next
        if he == dc:
            break  # we've completed the cycle

    # rewire da and bd to have twins of ac and cb respectively
    prep.repair_he_twins.append((da, ac))
    prep.repair_he_twins.append((bd, cb))

    # repair da and bd edges
    prep.repair_he_edges.append((da, ac.edge))
    prep.repair_he_edges.append((bd, cb.edge))

    # check if we broke ac.edge.halfedge
    if ac.edge.halfedge == ca:
        prep.repair_edge_hes.append((ac.edge, ac))

    # check if we broke cb.edge.halfedge
    if cb.edge.halfedge == bc:
        prep.repair_edge_hes.append((cb.edge, cb))

    # check if we broke c.halfedge
    if c.halfedge in (ca, cd):
        prep.repair_vert_hes.append((c, da))

    # check if we broke a.halfedge
    if a.halfedge == ad:
        prep.repair_vert_hes.append((a, ac))

    # check if we broke b.halfedge
    if b.halfedge == bc:
        prep.repair_vert_hes.append((b, bd))

    return prep


# TODO: P6 -- complete this
def do_collapse(prep: CollapsePrep, mesh: Mesh):
    """
    The second stage of edge-collapse.

    This function should implement all of the operations described in the
    `CollapsePrep` data-class (defined above). Ideally, this function should
    not need to traverse the mesh's topology at all, as all traversal should
    be handled by prepare_collapse().

    This should modify the mesh's topology and vertices coords inplace.
    (You should not need to create any new Primitives!)

    To delete primitives, for instance, a halfedge with index halfedge_id, use
        del mesh.topology.halfedges[halfedge_id]
    and similarly for other primitive types.
    """
    # TODO write your code here and replace this raise
    for he0, he1 in prep.repair_he_twins:
        he0.twin = he1
        he1.twin = he0

    for he0, he1 in prep.repair_he_nexts:
        he0.next = he1

    for he, edge in prep.repair_he_edges:
        he.edge = edge

    for he, vertex in prep.repair_he_verts:
        he.vertex = vertex

    for he, face in prep.repair_he_faces:
        he.face = face

    for edge, he in prep.repair_edge_hes:
        edge.halfedge = he

    for vertex, he in prep.repair_vert_hes:
        vertex.halfedge = he

    for face, he in prep.repair_face_hes:
        face.halfedge = he

    v0, v1 = prep.merge_verts
    m = np.mean(list(map(mesh.get_3d_pos, prep.merge_verts)), axis=0)
    mesh.vertices[v0.index] = m

    for he in prep.del_hes:
        del mesh.topology.halfedges[he.index]

    for edge in prep.del_edges:
        del mesh.topology.edges[edge.index]

    for vertex in prep.del_verts:
        del mesh.topology.vertices[vertex.index]

    for face in prep.del_faces:
        del mesh.topology.faces[face.index]


class EdgeCollapseWithLink(MeshEdit):
    def __init__(self, mesh: Mesh, e_id: int):
        self.mesh = mesh
        self.e_id = e_id
        self.link_cond = link_condition(self.mesh, self.e_id)
        if self.link_cond:
            self.prep = prepare_collapse(self.mesh, self.e_id)

    def apply(self):
        if not self.link_cond:
            print(f"Collapse is not doable, does not satisfy link condition")
            return
        return do_collapse(self.prep, self.mesh)


# TODO: Extra credit -- complete this
def link_condition(mesh: Mesh, e_id: int) -> bool:
    """
    Return whether the mesh and the specified edge satisfy the link condition.
    """
    topology = mesh.topology
    e = topology.edges[e_id]
    # TODO write your code here and replace this raise and return
    cd = e.halfedge
    dc = cd.twin

    c = cd.vertex
    d = dc.vertex

    c_neighbors = {v.index for v in c.adjacentVertices()}
    d_neighbors = {v.index for v in d.adjacentVertices()}

    return len(c_neighbors.intersection(d_neighbors)) == 2
