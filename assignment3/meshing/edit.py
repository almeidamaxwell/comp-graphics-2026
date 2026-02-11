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
        raise NotImplementedError("TODO (P5)")


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
    raise NotImplementedError("TODO (P6)")


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
    raise NotImplementedError("TODO (P6)")


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
    raise NotImplementedError("TODO (EC)")
