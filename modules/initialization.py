import taichi as ti
import yaml
import igl
import numpy as np

def init_params():
    """Load in parameters specified in config.yaml"""
    with open("config.yaml", 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_mesh(mesh_path):
    """Loads in mesh (obj, off, stl, wrl, ply, mesh) at location mesh_path
        and converts it to triangle mesh if necessary
       Output:
        * x: mesh vertices; shape (num_vertices, 3); used in simulation
        * v: vertex velocities; shape (num_vertices, 3); used in simulation
        * vertices: mesh vertices; shape (num_vertices,); used in rendering
        * indices: mesh triangle indices;  shape (num_triangles,); used in rendering"""

    v_np,f = igl.read_triangle_mesh(mesh_path)

    f_np = f.flatten()

    # Convert vertices and indices to ti.Vector.field
    n_vertices = len(v_np)
    n_triangles = int(len(f_np) // 3)

    x = ti.field(ti.f64, shape=v_np.shape)
    x.from_numpy(v_np)

    v = ti.field(ti.f64, shape=v_np.shape)
    v.from_numpy(np.zeros(v_np.shape))

    # For the gui (1D arrays)
    vertices = ti.Vector.field(3, dtype=ti.f64, shape=n_vertices)
    indices = ti.field(int, shape=n_triangles * 3)
    indices.from_numpy(f_np)

    return x, v, vertices, indices

