import taichi as ti
import taichi.math as math
@ti.func
def edge_J(x1, y1, z1, x2, y2, z2, e_bar) -> ti.types.vector(6, float):
 x0 = x1 - x2
 x3 = y1 - y2
 x4 = z1 - z2
 x5 = math.sqrt(x0**2 + x3**2 + x4**2)
 x6 = 2*(-1 + x5/e_bar)/x5
 x7 = x0*x6
 x8 = x3*x6
 x9 = x4*x6
 dx1 = x7
 dy1 = x8
 dz1 = x9
 dx2 = -x7
 dy2 = -x8
 dz2 = -x9
 return dx1, dy1, dz1, dx2, dy2, dz2, 