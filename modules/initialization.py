import taichi as ti
import yaml
import igl
import numpy as np

def init_params():
    """Load in parameters specified in config.yaml"""
    with open("config.yaml", 'r') as file:
        params = yaml.safe_load(file)
    return params

@ti.kernel
def copy_vector(vertices:ti.template(), x:ti.template()):
    for i, j in ti.ndrange(*x.shape):
        vertices[i][j] = x[i, j]

def init_vector(np_v, dim, dtype, shape):
    field = ti.field(dtype, shape=np_v.shape)
    field.from_numpy(np_v)
    vector_field = ti.Vector.field(dim, dtype=dtype, shape=shape)
    copy_vector(vector_field, field)
    return vector_field


def load_mesh(mesh_path):
    """Loads in mesh (obj, off, stl, wrl, ply, mesh) at location mesh_path
        and converts it to triangle mesh if necessary
       Output:
        * vertices_undef: mesh vertices; shape (n_vertices,) - inital vertices
        * vertices: mesh vertices; shape (n_vertices,)
        * indices: mesh triangle indices;  shape (n_triangles * 3,); used in rendering
        * t_indices: mesh triangle indices;  shape (n_triangles,)
        * e_indices: mesh triangle indices;  shape (n_edges)
        * adj_t_indices: mesh triangle indices;  shape (n_adj_triangles,)"""

    v_np,f = igl.read_triangle_mesh(mesh_path)
    edge_indices, adj_tri_indices = load_indices(f)

    f_np = f.flatten()

    # Convert vertices and indices to ti.Vector.field
    n_vertices = len(v_np)
    n_triangles = len(f)
    n_edges = len(edge_indices)
    n_adj_triangles = len(adj_tri_indices)

    #Init vertices, have to be ndarray for solver
    vertices_gui = init_vector(v_np, dim=3, dtype=ti.float32, shape=n_vertices) #needed for gui
    # vertices_undef = init_vector(v_np, dim=3, dtype=ti.float32, shape=n_vertices)
    vertices = ti.ndarray(dtype=ti.math.vec3, shape=n_vertices)
    vertices.from_numpy(v_np)
    vertices_undef = ti.ndarray(dtype=ti.math.vec3, shape=n_vertices)
    vertices_undef.from_numpy(v_np)


    # Init vel
    vels_np = np.zeros(v_np.shape)
    vels = init_vector(vels_np, dim=3, dtype=ti.float32, shape=n_vertices)

    #Init edge indices
    e_indices = init_vector(edge_indices, dim=2, dtype=int, shape=n_edges)

    #Init adjacent triangle indices
    adj_t_indices = init_vector(adj_tri_indices, dim=4, dtype=int, shape=n_adj_triangles)

    #Init triangle indices
    t_indices = init_vector(f, dim=3, dtype=int, shape=n_triangles)

    # For the gui (1D arrays)
    gui_indices = ti.field(int, shape=n_triangles * 3)
    gui_indices.from_numpy(f_np)

    return vertices_undef, vertices, vels, vertices_gui, gui_indices, t_indices, e_indices, adj_t_indices

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


    adj_tris, rel_edge = igl.triangle_triangle_adjacency(tri_indices)

    #(tri_id1, tri_id2, edge_id)
    adj_tri_indices = []
    for tri_idx, (tri0, tri1, tri2) in enumerate(adj_tris):
        if tri0 != -1 and tri0 > tri_idx:
            adj_tri_indices.append([
                tri_indices[tri_idx][0],
                tri_indices[tri_idx][1],
                tri_indices[tri_idx][2],
                tri_indices[tri0][(rel_edge[tri_idx][0]+2)%3]])
        if tri1 != -1 and tri1 > tri_idx:
            adj_tri_indices.append([
                tri_indices[tri_idx][1],
                tri_indices[tri_idx][2],
                tri_indices[tri_idx][0],
                tri_indices[tri1][(rel_edge[tri_idx][1]+2)%3]])
        if tri2 != -1 and tri2 > tri_idx:
            adj_tri_indices.append([
                tri_indices[tri_idx][2],
                tri_indices[tri_idx][0],
                tri_indices[tri_idx][1],
                tri_indices[tri2][(rel_edge[tri_idx][2]+2)%3]])

    adj_tri_indices = np.unique(adj_tri_indices,axis=0)
    return edge_indices, adj_tri_indices

def init_rest_edge_lengths():
    """
    Returns list of edge lengths of the undeformed edges
    with rest_edge_lengths[i] = rest edge length of edge i
    """
    #TODO
    pass

def init_rest_triangle_areas():
    """
    Returns list of triangle areas of the undeformed triangles
    with rest_triangle_areas[i] = rest triangle area of triangle i
    """
    #TODO
    pass

def init_rest_dihedral_angles():
    """
    Returns list of dihedral angles of the undeformed adjacent triangles
    with rest_dihedral_angles[i] = rest dihedral angle of adjacent triangles i
    """
    #TODO
    pass

def init_rest_heights():
    """
    Returns list of heights of the undeformed adjacent triangles
    with rest_heights[i] = rest height of adjacent triangles i
    """
    #TODO
    pass