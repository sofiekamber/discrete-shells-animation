import taichi as ti
import taichi.math as math
@ti.func
def area_J(x1, y1, z1, x2, y2, z2, x3, y3,z3, A_bar):
 x0 = -y3
 x4 = x0 + y2
 x5 = x1 - x2
 x6 = x0 + y1
 x7 = -x3
 x8 = x1 + x7
 x9 = y1 - y2
 x10 = -x5*x6 + x8*x9
 x11 = -z3
 x12 = x11 + z2
 x13 = x11 + z1
 x14 = z1 - z2
 x15 = -x13*x5 + x14*x8
 x16 = -x13*x9 + x14*x6
 x17 = math.sqrt(x10**2 + x15**2 + x16**2)
 x18 = (1/2)*(-2 + x17/A_bar)/x17
 x19 = x2 + x7
 dx1 = -x18*(x10*x4 + x12*x15)
 dy1 = x18*(x10*x19 - x12*x16)
 dz1 = x18*(x15*x19 + x16*x4)
 dx2 = x18*(x10*x6 + x13*x15)
 dy2 = -x18*(x10*x8 - x13*x16)
 dz2 = -x18*(x15*x8 + x16*x6)
 dx3 = -x18*(x10*x9 + x14*x15)
 dy3 = x18*(x10*x5 - x14*x16)
 dz3 = x18*(x15*x5 + x16*x9)
 return dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3, 