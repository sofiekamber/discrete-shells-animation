import taichi as ti
import taichi.math as math

@ti.func
def edge_length(v0, v1):
    return (v1 - v0).norm()

@ti.func
def triangle_area(v0, v1, v2):
    e0 = v1 - v0
    e1 = v2 - v0
    return 0.5 * (e0.cross(e1)).norm()


@ti.func
def triangle_normal(v0, v1, v2):
    e0 = v1 - v0
    e1 = v2 - v0
    return (e0.cross(e1)).normalized()


@ti.func
def dihedral_angle(normal0, normal1):
    dotProduct = normal0.dot(normal1)
    #// Ensure the dot product is within valid range [-1, 1] to avoid NaN in arccos
    dotProduct = math.max(-1.0, math.min(1.0, dotProduct))
    angleRad = math.acos(dotProduct)
    angleDeg = angleRad * 180.0 / math.pi

    return angleRad


@ti.func
def height(t0_area, t1_area, e_length):
    #1/6 * (height_t0 + height_t1) sharing e
    height_t0 = 2 * t0_area / e_length
    height_t1 = 2 * t1_area / e_length
    return (1/6) * (height_t0 + height_t1)