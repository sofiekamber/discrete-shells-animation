import taichi as ti
import numpy as np
import taichi.lang.ops
import taichi.math as math
import energy_helper_funcs as ef

ti.init(arch=ti.cpu,debug=True)


@ti.kernel
def test_edge_length(x0:ti.math.vec3, x1:ti.math.vec3, gt_e:float):
    e = ef.edge_length(x0,x1)
    success = e == gt_e
    if success:
        print("[PASSED] Edge Length")
    else:
        print("[FAILED] Edge Length")

@ti.kernel
def test_triangle_area(x0:ti.math.vec3, x1:ti.math.vec3, x2:ti.math.vec3, gt_a:float):
    a = ef.triangle_area(x0,x1,x2)
    success = a == gt_a
    if success:
        print("[PASSED] Triangle Area")
    else:
        print("[FAILED] Triangle Area")


@ti.kernel
def test_normal(x0:ti.math.vec3, x1:ti.math.vec3, x2:ti.math.vec3, gt_n:ti.math.vec3):
    n = ef.triangle_normal(x0,x1,x2)
    success = math.round(n[0],float) == math.round(gt_n[0],float) \
              and math.round(n[1],float) == math.round(gt_n[1],float) \
              and math.round(n[2],float) == math.round(gt_n[2],float)
    if success:
        print("[PASSED] Normal")
    else:
        print("[FAILED] Normal")


@ti.kernel
def test_dihedral_angle(x0:ti.math.vec3, x1:ti.math.vec3, x2:ti.math.vec3,
                        x0_:ti.math.vec3, x1_:ti.math.vec3, x2_:ti.math.vec3, gt_angle:float):
    normal0 = ef.triangle_normal(x0,x1,x2)
    normal1 = ef.triangle_normal(x0_,x1_,x2_)
    angle = ef.dihedral_angle(normal0,normal1)
    success = angle == gt_angle
    if success:
        print("[PASSED] Dihedral Angle")
    else:
        print("[FAILED] Dihedral Angle")

@ti.kernel
def test_height(x0:ti.math.vec3, x1:ti.math.vec3, x2:ti.math.vec3,
                x0_:ti.math.vec3, x1_:ti.math.vec3, x2_:ti.math.vec3,
                x_e0:ti.math.vec3 , x_e1:ti.math.vec3, gt_height:float):
    t0_area = ef.triangle_area(x0,x1,x2)
    t1_area = ef.triangle_area(x0_,x1_,x2_)
    e_length = ef.edge_length(x_e0, x_e1)
    height = ef.height(t0_area, t1_area, e_length)
    success = height == gt_height
    if success:
        print("[PASSED] Height")
    else:
        print("[FAILED] Height")


if __name__ == "__main__":
    v = np.array([[0.688176, 0.244554, -0.656993], [0.673000, 1.000000, 0.739646], [-0.386252, 0.371472, 1.707274], [-0.366183, 1.000000, -0.397935]], dtype=np.float32)
    vertices = ti.ndarray(dtype=ti.math.vec3, shape=v.shape[0])
    vertices.from_numpy(v)

    t0_indices = [1,3,2]
    t1_indices = [1,0,3]

    print("Testing...")
    #Edge test
    v0_np,v1_np,v2_np = v[[1,3,2]]
    v0, v1, v2 = ti.math.vec3(v0_np), ti.math.vec3(v1_np), ti.math.vec3(v2_np)

    v0_np_, v1_np_, v2_np_ = v[[1,0,3]]
    v0_, v1_, v2_ = ti.math.vec3(v0_np_), ti.math.vec3(v1_np_), ti.math.vec3(v2_np_)

    edge_length = np.linalg.norm(v[t0_indices[1]] - v[t0_indices[0]])
    test_edge_length(v0, v1, edge_length)

    #Area test
    side1 = v[t0_indices[1]] - v[t0_indices[0]]
    side2 = v[t0_indices[2]] - v[t0_indices[0]]
    cross_product = np.cross(side1, side2)
    area = 0.5 * np.linalg.norm(cross_product)
    test_triangle_area(v0, v1, v2, area)

    #Normal test
    normal = np.cross(side1, side2)
    normal /= np.linalg.norm(normal)
    test_normal(v0, v1, v2,  ti.math.vec3(normal))

    #Angle test
    side1 = v[t1_indices[1]] - v[t1_indices[0]]
    side2 = v[t1_indices[2]] - v[t1_indices[0]]
    normal1 = np.cross(side1, side2)
    normal1 /= np.linalg.norm(normal1)
    dotProduct = normal.dot(normal1)
    #Ensure the dot product is within valid range [-1, 1] to avoid NaN in arccos
    dotProduct = max(-1.0, min(1.0, dotProduct))
    angleRad = np.arccos(dotProduct)
    test_dihedral_angle(v0, v1, v2, v0_, v1_, v2_, angleRad)

    #Height test
    t0_area = area
    cross_product = np.cross(side1, side2)
    t1_area = 0.5 * np.linalg.norm(cross_product)
    e_length = np.linalg.norm(v1-v0)
    height_t0 = 2 * t0_area / e_length
    height_t1 = 2 * t1_area / e_length
    height = (1 / 6) * (height_t0 + height_t1)
    test_height(v0, v1, v2, v0_, v1_, v2_,v0, v1,height)



