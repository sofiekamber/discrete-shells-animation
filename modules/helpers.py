import taichi as ti

@ti.func
def id_x(i: int) -> int:
    return 3*i

@ti.func
def id_y(i: int) -> int:
    return 3*i + 1

@ti.func 
def id_z(i:int) -> int:
    return 3*i + 2

@ti.func 
def global_idx(i:int) -> ti.types.vector(3, int):
    return id_x(i), id_y(i), id_z(i)

@ti.func
def global_idx_2(i0:int, i1:int) -> ti.types.vector(6,int):
    return id_x(i0), id_y(i0), id_z(i0), id_x(i1), id_y(i1), id_z(i1)

@ti.func
def get_vertex_coords(i: int, V: ti.template()) -> ti.types.vector(3, float):
    i_x, i_y, i_z = global_idx(i)
    return ti.Vector([ V[i_x], V[i_y], V[i_z]])
