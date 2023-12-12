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

rest_adj_tri_metadata = ti.Vector.field(n=3, dtype=ti.float32, shape=n_adj_triangles)
init.init_rest_adj_tri_metadata(adj_t_ids, x, rest_adj_tri_metadata)


"""Run gui"""
window = ti.ui.Window("Discrete Shells", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0

dt = params["dt"]
substeps = int(1 / 60 // dt)

M_inverse = ti.linalg.SparseMatrixBuilder(num_rows=n_vertices, num_cols=n_vertices, max_num_triplets=100)

x[0] = 2.0
print(x.to_numpy())
while window.running:
    if current_t > 1.5:
        # Reset
        current_t = 0

    for i in range(substeps):
        sim.newmark_integration(x_i=x, v_i =v,delta_t=dt, beta=params["beta"], gamma=params["gamma"],
                                e_ids=e_ids, rest_edge_lengths=rest_edge_lengths, n_edges=n_edges, t_ids=t_ids,
                                A_bars=rest_triangle_areas, n_tris=n_triangles, adj_t_ids=adj_t_ids,
                                rest_adj_tri_metadata=rest_adj_tri_metadata, n_adj_triangles=n_adj_triangles)
        print("-----------")
        sim.update_vertices(x,vertices_gui)
        current_t += dt


    camera.position(3.0, 3.0, 4)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(5, 5, 5), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices_gui,
               indices=gui_indices,
               # per_vertex_color=colors,
               two_sided=True)

    canvas.scene(scene)
    window.show()
