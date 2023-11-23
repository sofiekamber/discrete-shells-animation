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
    edge_indices, adj_tri_indices = load_indices(f)

    f_np = f.flatten()

    # Convert vertices and indices to ti.Vector.field
    n_vertices = len(v_np)
    n_triangles = int(len(f_np) // 3)

    x = ti.field(ti.float32, shape=v_np.shape)
    x.from_numpy(v_np)

    v = ti.field(ti.float32, shape=v_np.shape)
    v.from_numpy(np.zeros(v_np.shape))

    e_indices = ti.field(int, shape=edge_indices.shape)
    e_indices.from_numpy(edge_indices)

    adj_t_indices = ti.field(int, shape=adj_tri_indices.shape)
    adj_t_indices.from_numpy(adj_tri_indices)

    # For the gui (1D arrays)
    vertices = ti.Vector.field(3, dtype=ti.float32, shape=n_vertices)
    t_indices = ti.field(int, shape=n_triangles * 3)
    t_indices.from_numpy(f_np)

    return x, v, vertices, t_indices, e_indices, adj_t_indices

def load_indices(tri_indices):
    """Returns edge indices and adjacent triangles to iterate through
    triangle_triangle_adjacency of igl"""

    #TT #F by #3 adjacent matrix, the element i,j is the id of the triangle adjacent to the j edge of triangle i
    #TTi #F by #3 adjacent matrix, the element i,j is the id of edge of the triangle TT(i,j) that is adjacent with triangle i

    #Retrieve sorted edge indices
    edge_indices = igl.edges(tri_indices)

    #Retrieve adjacent triangle indices
    #face_edge_dict: (triangle_id, triangle_edge_id) = edge_id in edge_indices
    face_edge_dict={}
    for tri_idx, (v0,v1,v2) in enumerate(tri_indices):
        edge_id0 = np.where(np.all(edge_indices == sorted([v0,v1]), axis=1))[0][0]
        face_edge_dict[(tri_idx,0)] = edge_id0
        edge_id1 = np.where(np.all(edge_indices == sorted([v1, v2]), axis=1))[0][0]
        face_edge_dict[(tri_idx,1)] = edge_id1
        edge_id2 = np.where(np.all(edge_indices == sorted([v0, v2]), axis=1))[0][0]
        face_edge_dict[(tri_idx,2)] = edge_id2


    adj_tris, _ = igl.triangle_triangle_adjacency(tri_indices)

    #(tri_id1, tri_id2, edge_id)
    adj_tri_indices = []
    for tri_idx, (tri0, tri1, tri2) in enumerate(adj_tris):
        if tri0 != -1:
            adj_tri_indices.append(sorted([tri_idx,tri0]) + [face_edge_dict[(tri_idx,0)]])
        if tri1 != -1:
            adj_tri_indices.append(sorted([tri_idx,tri1]) + [face_edge_dict[(tri_idx,1)]])
        if tri2 != -1:
            adj_tri_indices.append(sorted([tri_idx,tri2]) + [face_edge_dict[(tri_idx,2)]])

    adj_tri_indices = np.unique(adj_tri_indices,axis=0)
    return edge_indices, adj_tri_indices