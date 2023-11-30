#Import initialzation module (mesh init)
import modules.initialization as init

#Load params
params = init.init_params() #e.g. params["dt"]

import taichi as ti
device = params["device"]
if device == 'cpu':
    ti.init(arch=ti.cpu,debug=True)
elif device == 'gpu':
    ti.init(arch=ti.gpu)
else:
    raise ValueError(f"{device} not supported")
#Import energy functions
import modules.energies as en

#Import simulation functions (Newmark integration, differentiation)
import modules.simulation as sim

"""Initialize"""


#Init mesh
x_init, x, v, vertices_gui, gui_indices, t_ids, e_ids, adj_t_ids = init.load_mesh(params["mesh_path"])

n_vertices = x.shape[0]
n_triangles = t_ids.shape[0]
n_edges = e_ids.shape[0]
n_adj_triangles = adj_t_ids.shape[0]

#Initialize rest
rest_edge_lengths = ti.Vector.field(1, dtype=ti.float32, shape=n_edges)
init.init_rest_edge_lengths(e_ids, x, rest_edge_lengths)

rest_triangle_areas = ti.Vector.field(1, dtype=ti.float32, shape=n_triangles)
init.init_rest_triangle_areas(t_ids, x, rest_triangle_areas)

rest_dihedral_angles= ti.Vector.field(1, dtype=ti.float32, shape=n_adj_triangles)
init.init_rest_dihedral_angles(adj_t_ids, x, rest_dihedral_angles)

rest_heights = ti.Vector.field(1, dtype=ti.float32, shape=n_adj_triangles)
init.init_rest_heights(adj_t_ids, x, rest_heights)

# helper tests
from modules.helpers import *
@ti.kernel 
def proj_test():
    test = global_idx_2(0,1)
    print(test)

proj_test()

#iterator tests
print(x.to_numpy())
print(t_ids.to_numpy())
x[0][0] = 1
print(x.to_numpy())


from modules.energy_iterators import populate_edge_jacobian

J = ti.ndarray(float, 3*n_vertices)
populate_edge_jacobian(J, e_ids, x, rest_edge_lengths, n_edges)
print(J.to_numpy())

from modules.energy_iterators import populate_edge_hessian

H = ti.linalg.SparseMatrixBuilder(3*n_vertices, 3*n_vertices)
populate_edge_hessian(H, e_ids, x, rest_edge_lengths, n_edges)
print(H.build())
