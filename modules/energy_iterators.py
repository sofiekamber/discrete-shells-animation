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
        x1, y1, z1 = vertices[v0_idx * 3], vertices[v0_idx * 3 + 1],vertices[v0_idx * 3 + 2]
        x2, y2, z2 = vertices[v1_idx * 3], vertices[v1_idx * 3 + 1],vertices[v1_idx * 3 + 2]
        e_bar = e_bars[i]

        dx1, dy1, dz1, dx2, dy2, dz2 = edge_J(x1, y1, z1, x2, y2, z2, e_bar)
        J[id_x(v0_idx)] += dx1[0]
        J[id_y(v0_idx)] += dy1[0]
        J[id_z(v0_idx)] += dz1[0]

        J[id_x(v1_idx)] += dx2[0]
        J[id_y(v1_idx)] += dy2[0]
        J[id_z(v1_idx)] += dz2[0]

# @ti.kernel
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
        e_bar = e_bars[i]
        v0_idx, v1_idx = e_ids[i]
        x1, y1, z1 = vertices[v0_idx * 3], vertices[v0_idx * 3 + 1], vertices[v0_idx * 3 + 2]
        x2, y2, z2 = vertices[v1_idx * 3], vertices[v1_idx * 3 + 1], vertices[v1_idx * 3 + 2]

        h_entries = edge_H(x1,y1,z1, x2,y2,z2, e_bar)
        h_indexes = global_idx_2(v0_idx, v1_idx)

        for out_i in ti.static(range(6)):
            first_d = h_indexes[out_i]
            for in_i in ti.static(range(6)):
                second_d = h_indexes[in_i]

                H[first_d, second_d] += h_entries[6*out_i + in_i][0]

@ti.kernel
def populate_area_jacobian(
        J: ti.types.ndarray(),
        t_ids: ti.template(),
        vertices: ti.types.ndarray(),
        A_bars: ti.template(),
        n_tris: int):
    """
    Populate the jacobian with the area energy contributions.

    Inputs:
    J:          jacobian to fill
    t_ids:      T x 3 per triangle vertex indices
    vertices:   3N x 1 unrolled vector of vertex coordinates
    A_bars:     T x 1 triangle areas at rest
    n_tris:     T
    """
    for i in range(n_tris):
        v1, v2, v3 = t_ids[i]
        A_bar = A_bars[i]

        x1, y1, z1 = get_vertex_coords(v1, vertices) 
        x2, y2, z2 = get_vertex_coords(v2, vertices) 
        x3, y3, z3 = get_vertex_coords(v3, vertices) 

        j_entries = area_J(x1, y1, z1, x2, y2, z2, x3, y3, z3, A_bar)
        j_indexes = global_idx(v1) + global_idx(v2) + global_idx(v3)

        for j in ti.static(range(9)):
            d = j_indexes[j]
            J[d] += j_entries[j][0]

@ti.kernel
def populate_area_hessian(
        H: ti.types.sparse_matrix_builder(),
        t_ids: ti.template(),
        vertices: ti.types.ndarray(),
        A_bars: ti.template(),
        n_tris: int):
    """
    Populate the hessian with the area energy contributions.

    Inputs:
    H:          hessian to fill
    t_ids:      T x 3 per triangle vertex indices
    vertices:   3N x 1 unrolled vector of vertex coordinates
    A_bars:     T x 1 triangle areas at rest
    n_tris:     T
    """
    for i in range(n_tris):
        v1, v2, v3 = t_ids[i]
        A_bar = A_bars[i]

        x1, y1, z1 = get_vertex_coords(v1, vertices) 
        x2, y2, z2 = get_vertex_coords(v2, vertices) 
        x3, y3, z3 = get_vertex_coords(v3, vertices) 

        h_entries = area_H(x1, y1, z1, x2, y2, z2, x3, y3, z3, A_bar)
        h_indexes = global_idx(v1) + global_idx(v2) + global_idx(v3)

        for j in ti.static(range(9)):
            first_d = h_indexes[j]
            for k in ti.static(range(9)):
                second_d = h_indexes[k]

                H[first_d, second_d] += h_entries[j*9 + k][0]

@ti.kernel
def populate_flex_jacobian(
        J: ti.types.ndarray(),
        adj_t_ids: ti.template(),
        vertices: ti.types.ndarray(),
        meta_data: ti.template(),
        n_adj_tris: int):

    """
    Populate the jacobian with the flex energy contributions.

    Inputs:
    J:          jacobian to fill
    adj_t_ids:  AT x 4 four ids for an adj_tri the first two define the joining edge 
    vertices:   3N x 1 unrolled vector of vertex coordinates
    meta_data: AT x 3 per adjacent triangles hinge edge length, dihaderal angle, 1/3 average height
    n_adj_tris: AT
    """
    for i in range(n_adj_tris):
        v1, v2, v3, v4 = adj_t_ids[i]
        e_bar, theta_bar, h_bar = meta_data[i]

        x1, y1, z1 = get_vertex_coords(v1, vertices) 
        x2, y2, z2 = get_vertex_coords(v2, vertices) 
        x3, y3, z3 = get_vertex_coords(v3, vertices) 
        x4, y4, z4 = get_vertex_coords(v4, vertices) 

        j_entries = flex_J(x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4, e_bar, theta_bar, h_bar)
        j_indexes = global_idx(v1) + global_idx(v2) + global_idx(v3) + global_idx(v4)

        for j in ti.static(range(12)):
            d = j_indexes[j]
            entry = j_entries[j]
            if not ti.math.isnan(entry):
                J[d] += entry

            
@ti.kernel
def populate_flex_hessian(
        H: ti.types.sparse_matrix_builder(),
        adj_t_ids: ti.template(),
        vertices: ti.types.ndarray(),
        meta_data: ti.template(),
        n_adj_tris: int):

    """
    Populate the hessian with the flex energy contributions.

    Inputs:
    H:          hessian to fill
    adj_t_ids:  AT x 4 four ids for an adj_tri the first two define the joining edge 
    vertices:   3N x 1 unrolled vector of vertex coordinates
    meta_data: AT x 3 per adjacent triangles hinge edge length, dihaderal angle, 1/3 average height
    n_adj_tris: AT
    """
    for i in range(n_adj_tris):
        v1, v2, v3, v4 = adj_t_ids[i]
        e_bar, theta_bar, h_bar = meta_data[i]

        x1, y1, z1 = get_vertex_coords(v1, vertices) 
        x2, y2, z2 = get_vertex_coords(v2, vertices) 
        x3, y3, z3 = get_vertex_coords(v3, vertices) 
        x4, y4, z4 = get_vertex_coords(v4, vertices) 

        h_entries = flex_H(x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4, e_bar, theta_bar, h_bar)
        h_indexes = global_idx(v1) + global_idx(v2) + global_idx(v3) + global_idx(v4)

        for j in ti.static(range(12)):
            first_d = h_indexes[j]
            for k in ti.static(range(12)):
                second_d = h_indexes[k]
                entry = h_entries[j*12 + k]
                if not ti.math.isnan(entry):
                    H[first_d, second_d] += entry
