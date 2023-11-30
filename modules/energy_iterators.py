import taichi as ti
from modules.helpers import *

from sym_diff.edge_J import edge_J
from sym_diff.edge_H import edge_H
from sym_diff.area_J import area_J
from sym_diff.area_H import area_H
from sym_diff.flex_J import flex_J
from sym_diff.flex_H import flex_H

@ti.kernel
def populate_edge_jacobian(
        J: ti.types.ndarray(),
        e_ids: ti.template(),
        vertices: ti.types.ndarray(),
        e_bars: ti.template(),
        n_edges: int):
    """
    Populate the jacobian with the jacobina of all edge energies
    """
    for i in range(n_edges):
        v0_idx, v1_idx = e_ids[i]
        x1, y1, z1 = vertices[v0_idx]
        x2, y2, z2 = vertices[v1_idx]
        e_bar = e_bars[i]

        dx1, dy1, dz1, dx2, dy2, dz2 = edge_J(x1, y1, z1, x2, y2, z2, e_bar)
        J[id_x(v0_idx)] += dx1[0]
        J[id_y(v0_idx)] += dy1[0]
        J[id_z(v0_idx)] += dz1[0]

        J[id_x(v1_idx)] += dx2[0]
        J[id_y(v1_idx)] += dy2[0]
        J[id_z(v1_idx)] += dz2[0]

@ti.kernel
def populate_edge_hessian(
        H: ti.types.sparse_matrix_builder(),
        e_ids: ti.template(),
        vertices: ti.types.ndarray(),
        e_bars: ti.template(),
        n_edges: int):
    """
    Populate the hessian with the edge energy contributions.
    """
    for i in range(n_edges):
        v1, v2 = e_ids[i]
        e_bar = e_bars[i]
        x1, y1, z1 = vertices[v1]
        x2, y2, z2 = vertices[v2]

        h_entries = edge_H(x1,y1,z1, x2,y2,z2, e_bar)
        h_indexes = global_idx_2(v1, v2)

        for out_i in ti.static(range(6)):
            first_d = h_indexes[out_i]
            for in_i in ti.static(range(6)):
                second_d = h_indexes[in_i]

                H[first_d, second_d] += h_entries[6*out_i + in_i][0]

        
        


