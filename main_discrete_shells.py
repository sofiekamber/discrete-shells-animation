#Import initialzation module (mesh init)
import modules.initialization as init

#Load params
params = init.init_params() #e.g. params["dt"]

import taichi as ti
device = params["device"]
if device == 'cpu':
    ti.init(arch=ti.cpu)
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
x_init, x, v, gui_indices, t_ids, e_ids, adj_t_ids = init.load_mesh(params["mesh_path"])

n_vertices = x.shape[0]
n_triangles = t_ids.shape[0]
n_edges = e_ids.shape[0]
n_adj_triangles = adj_t_ids.shape[0]


"""Run gui"""
window = ti.ui.Window("Discrete Shells", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

current_t = 0.0

dt = params["dt"]
substeps = int(1 / 60 // dt)



while window.running:
    if current_t > 1.5:
        # Reset
        current_t = 0

    for i in range(substeps):
        #sim.newmark_integration()
        current_t += dt


    camera.position(3.0, 3.0, 4)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(5, 5, 5), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(x,
               indices=gui_indices,
               # per_vertex_color=colors,
               two_sided=True)

    canvas.scene(scene)
    window.show()
