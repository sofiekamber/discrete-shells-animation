import taichi as ti
import taichi.math as math
@ti.func
def flex_J(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, e_bar, theta_bar, h_bar):
 x0 = -y3
 x5 = x0 + y2
 x6 = x1 - x2
 x7 = -y4
 x8 = x7 + y1
 x9 = -x4
 x10 = x1 + x9
 x11 = y1 - y2
 x12 = x10*x11 - x6*x8
 x13 = x12*x5
 x14 = x7 + y2
 x15 = x0 + y1
 x16 = -x3
 x17 = x1 + x16
 x18 = x11*x17 - x15*x6
 x19 = x14*x18
 x20 = -z3
 x21 = x20 + z2
 x22 = -z4
 x23 = x22 + z1
 x24 = z1 - z2
 x25 = x10*x24 - x23*x6
 x26 = x21*x25
 x27 = x22 + z2
 x28 = x20 + z1
 x29 = x17*x24 - x28*x6
 x30 = x27*x29
 x31 = x12*x29 - x18*x25
 x32 = -x11*x23 + x24*x8
 x33 = -x11*x28 + x15*x24
 x34 = x12*x33 - x18*x32
 x35 = x25*x33 - x29*x32
 x36 = x31**2 + x34**2 + x35**2
 x37 = math.sqrt(x36)
 x38 = x12*x18 + x25*x29 + x32*x33
 x39 = 1/(x36 + x38**2)
 x40 = x37*x39
 x41 = x32*x5
 x42 = x14*x33
 x43 = x21*x32
 x44 = -x27*x33
 x45 = x25*x5
 x46 = x18*x27
 x47 = x14*x29
 x48 = x12*x21
 x49 = 1/x37
 x50 = x38*x39*x49
 x51 = 2*e_bar*(theta_bar - math.atan2(x37, -x38))/h_bar
 x52 = x16 + x2
 x53 = x12*x52
 x54 = x2 + x9
 x55 = x18*x54
 x56 = x25*x52
 x57 = x29*x54
 x58 = x32*x52 - x33*x54
 x59 = x12*x15
 x60 = x18*x8
 x61 = x25*x28
 x62 = x23*x29
 x63 = x15*x32
 x64 = x33*x8
 x65 = x28*x32
 x66 = -x23*x33
 x67 = x15*x25
 x68 = x18*x23
 x69 = x29*x8
 x70 = x12*x28
 x71 = x12*x17
 x72 = x10*x18
 x73 = x17*x25
 x74 = x10*x29
 x75 = -x10*x33 + x17*x32
 x76 = x11*x12
 x77 = x24*x25
 x78 = x11*x32
 x79 = x24*x32
 x80 = x11*x25
 x81 = x12*x24
 x82 = x12*x6
 x83 = x25*x6
 x84 = x32*x6
 x85 = x11*x18
 x86 = x24*x29
 x87 = x11*x33
 x88 = x24*x33
 x89 = x11*x29
 x90 = x18*x24
 x91 = x18*x6
 x92 = x29*x6
 x93 = x33*x6
 dx1 = x51*(x40*(x13 + x19 + x26 + x30) + x50*(x31*(x45 + x46 - x47 - x48) + x34*(x41 - x42) + x35*(x43 + x44)))
 dy1 = -x51*(x40*(-x43 + x44 + x53 + x55) + x50*(x31*(x56 - x57) + x34*(-x46 + x48 + x58) + x35*(x26 - x30)))
 dz1 = x51*(-x40*(x41 + x42 + x56 + x57) + x50*(x31*(x53 - x55) + x34*(x13 - x19) - x35*(-x45 + x47 + x58)))
 dx2 = -x51*(x40*(x59 + x60 + x61 + x62) + x50*(x31*(x67 + x68 - x69 - x70) + x34*(x63 - x64) + x35*(x65 + x66)))
 dy2 = x51*(x40*(-x65 + x66 + x71 + x72) + x50*(x31*(x73 - x74) + x34*(-x68 + x70 + x75) + x35*(x61 - x62)))
 dz2 = -x51*(-x40*(x63 + x64 + x73 + x74) + x50*(x31*(x71 - x72) + x34*(x59 - x60) - x35*(-x67 + x69 + x75)))
 dx3 = x51*(x40*(x76 + x77) + x50*(x31*(x80 - x81) + x34*x78 + x35*x79))
 dy3 = -x51*(x40*(-x79 + x82) + x50*(x31*x83 + x34*(x81 + x84) + x35*x77))
 dz3 = x51*(x38*x39*x49*(x31*x82 + x34*x76 - x35*(-x80 + x84)) - x40*(x78 + x83))
 dx4 = -x51*(x38*x39*x49*(x31*(x89 - x90) + x34*x87 + x35*x88) - x40*(x85 + x86))
 dy4 = x51*(x38*x39*x49*(x31*x92 + x34*(x90 + x93) + x35*x86) - x40*(-x88 + x91))
 dz4 = -x51*(x40*(x87 + x92) + x50*(x31*x91 + x34*x85 - x35*(-x89 + x93)))
 return dx1, dy1, dz1, dx2, dy2, dz2, dx3, dy3, dz3, dx4, dy4, dz4, 