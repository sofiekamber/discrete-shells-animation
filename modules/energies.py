"""Implementation of Discrete Shells - Total Engergy"""
import taichi as ti

@ti.kernel
def stretch_edge(k_stretch:ti.f64):
    """This function computes the energy E_L resulting from change of edge lengths"""
    pass

@ti.kernel
def stretch_area(k_area:ti.f64):
    """This function computes the energy E_A resulting from change of triangle areas"""
    pass


@ti.kernel
def bending_angles(k_bend:ti.f64):
    """This function computes the bending E_B resulting from change of dihedral angles"""
    pass