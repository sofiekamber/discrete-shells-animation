import taichi as ti
ti.init(arch=ti.cpu)

#Import initialzation module (mesh init)
import modules.initialization as init

#Import energy functions
import modules.energies as en

#Import simulation functions (Newmark integration, differentiation)
import modules.simulation as sim


"""Initialize"""
#Load params
params = init.init_params() #e.g. params["dt"]

#Init mesh
x, v, vertices, indices = init.load_mesh(params["mesh_path"])

#Access triangle vertices via tri_index indices[tri_id * 3 + v_id]; v_id is index relative to triangle (in [0,1,2])

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
    sim.update_vertices(vertices,x)


    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               # per_vertex_color=colors,
               two_sided=True)

    canvas.scene(scene)
    window.show()
