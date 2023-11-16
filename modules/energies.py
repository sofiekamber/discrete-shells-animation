"""Implementation of Discrete Shells - Total Engergy"""
import taichi as ti

@ti.kernel
def stretch_edge():
    """This function computes the energy E_L resulting from change of edge lengths"""
    pass

@ti.kernel
def stretch_area():
    """This function computes the energy E_A resulting from change of triangle areas"""
    pass


@ti.kernel
def bending_angles():
    """This function computes the bending E_B resulting from change of dihedral angles"""
    pass