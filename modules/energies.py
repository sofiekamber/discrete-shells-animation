"""Implementation of Discrete Shells - Total Engergy"""
import taichi as ti

@ti.kernel
def stretch_edge(k_stretch:ti.f64):
    """This function computes the energy E_L resulting from change of edge lengths"""
    pass

@ti.func
def tri_area(x, id0, id1, id2):
    """Returns triangle area of given vertices
    """
    term1 = ((x[id0, 1] - x[id1, 1]) * (x[id0, 2] - x[id2, 2])) - (
            (x[id0, 2] - x[id1, 2]) * (x[id0, 1] - x[id2, 1]))
    term2 = ((x[id0, 2] - x[id1, 2]) * (x[id0, 0] - x[id2, 0])) - (
            (x[id0, 0] - x[id1, 0]) * (x[id0, 2] - x[id2, 2]))
    term3 = ((x[id0, 0] - x[id1, 0]) * (x[id0, 1] - x[id2, 1])) - (
            (x[id0, 1] - x[id1, 1]) * (x[id0, 0] - x[id2, 0]))

    area = 0.5 * ti.sqrt(ti.pow(term1, 2) + ti.pow(term2, 2) + ti.pow(term3, 2))
    return area

@ti.kernel
def stretch_area(x_undef:ti.template(), x_def:ti.template(), indices:ti.template(), num_triangles: int, k_area:ti.f64):
    """This function computes the energy E_A resulting from change of triangle areas"""
    E_A = 0.0
    for tri_id in range(num_triangles):
        #compute area of each triangle
        id0 = indices[tri_id * 3 + 0]
        id1 = indices[tri_id * 3 + 1]
        id2 = indices[tri_id * 3 + 2]
        area_undef = tri_area(x_undef, id0, id1, id2)
        area_def = tri_area(x_def, id0, id1, id2)
        E_A += area_undef * ti.pow(((area_def/area_undef) - 1),2)

    return k_area * E_A


@ti.kernel
def bending_angles(k_bend:ti.f64):
    """This function computes the bending E_B resulting from change of dihedral angles"""
    pass