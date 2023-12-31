import taichi as ti
import yaml
import igl
import numpy as np
import modules.energy_helper_funcs as ef
from modules.helpers import get_vertex_coords

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
        * vels: vertex velocities; shape (n_vertices,) - initial vertex velocities set to 0
        * vertices_gui: mesh vertices; used in rendering
        * gui_indices: mesh triangle indices;  shape (n_triangles * 3,); used in rendering
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
    vertices = ti.ndarray(dtype=ti.float32, shape=3 * n_vertices)
    vertices.from_numpy(v_np.flatten())
    vertices_undef = ti.ndarray(dtype=ti.float32, shape=3 * n_vertices)
    vertices_undef.from_numpy(v_np.flatten())


    # Init vel
    vels_np = np.zeros(3 * n_vertices)
    vels = ti.ndarray(dtype=ti.float32, shape=3 * n_vertices)
    vels.from_numpy(vels_np)

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

@ti.kernel
def init_rest_edge_lengths(edges: ti.template(), vertices:ti.types.ndarray(dtype=ti.float32, ndim=1),
                             rest_edge_lengths:ti.template()):
    """
    Returns list of edge lengths of the undeformed edges
    with rest_edge_lengths[i] = rest edge length of edge i
    """
    for i in range(edges.shape[0]):
        v0_idx, v1_idx = edges[i]
        v0_idx *= 3
        v1_idx *= 3
        v0 = ti.math.vec3(vertices[v0_idx],vertices[v0_idx + 1],vertices[v0_idx + 2])
        v1 = ti.math.vec3(vertices[v1_idx],vertices[v1_idx + 1],vertices[v1_idx + 2])
        rest_edge_lengths[i] = ef.edge_length(v0, v1)

@ti.kernel
def init_rest_triangle_areas(triangles: ti.template(), vertices:ti.types.ndarray(dtype=ti.float32, ndim=1),
                             rest_triangle_areas:ti.template()):
    """
    Returns list of triangle areas of the undeformed triangles
    with rest_triangle_areas[i] = rest triangle area of triangle i
    """
    for i in range(triangles.shape[0]):
        v0_idx, v1_idx, v2_idx = triangles[i]
        v0_idx *= 3
        v1_idx *= 3
        v2_idx *= 3
        v0 = ti.math.vec3(vertices[v0_idx], vertices[v0_idx + 1], vertices[v0_idx + 2])
        v1 = ti.math.vec3(vertices[v1_idx], vertices[v1_idx + 1], vertices[v1_idx + 2])
        v2 = ti.math.vec3(vertices[v2_idx], vertices[v2_idx + 1], vertices[v2_idx + 2])
        rest_triangle_areas[i] = ef.triangle_area(v0, v1, v2)


@ti.kernel
def init_rest_dihedral_angles(adj_triangles: ti.template(), vertices:ti.types.ndarray(dtype=ti.float32, ndim=1),
                              rest_dihedral_angles:ti.template()):
    """
    Returns list of dihedral angles of the undeformed adjacent triangles
    with rest_dihedral_angles[i] = rest dihedral angle of adjacent triangles i
    """
    for i in range(adj_triangles.shape[0]):
        v0_idx, v1_idx, v2_idx, v3_idx = adj_triangles[i]
        v0_idx *= 3
        v1_idx *= 3
        v2_idx *= 3
        v3_idx *= 3
        v0 = ti.math.vec3(vertices[v0_idx], vertices[v0_idx + 1], vertices[v0_idx + 2])
        v1 = ti.math.vec3(vertices[v1_idx], vertices[v1_idx + 1], vertices[v1_idx + 2])
        v2 = ti.math.vec3(vertices[v2_idx], vertices[v2_idx + 1], vertices[v2_idx + 2])
        v3 = ti.math.vec3(vertices[v3_idx], vertices[v3_idx + 1], vertices[v3_idx + 2])
        normal0 = ef.triangle_normal(v0, v1, v2)
        normal1 = ef.triangle_normal(v0, v3, v1)
        rest_dihedral_angles[i] = ef.dihedral_angle(normal0,normal1)
        # rest_dihedral_angles[i] = 0.0

@ti.kernel
def init_rest_heights(adj_triangles: ti.template(), vertices:ti.types.ndarray(dtype=ti.float32, ndim=1),
                      rest_heights:ti.template()):
    """
    Returns list of heights of the undeformed adjacent triangles
    with rest_heights[i] = rest height of adjacent triangles i
    """
    for i in range(adj_triangles.shape[0]):
        v0_idx, v1_idx, v2_idx, v3_idx = adj_triangles[i]
        v0_idx *= 3
        v1_idx *= 3
        v2_idx *= 3
        v3_idx *= 3
        v0 = ti.math.vec3(vertices[v0_idx], vertices[v0_idx + 1], vertices[v0_idx + 2])
        v1 = ti.math.vec3(vertices[v1_idx], vertices[v1_idx + 1], vertices[v1_idx + 2])
        v2 = ti.math.vec3(vertices[v2_idx], vertices[v2_idx + 1], vertices[v2_idx + 2])
        v3 = ti.math.vec3(vertices[v3_idx], vertices[v3_idx + 1], vertices[v3_idx + 2])
        t0_area = ef.triangle_area(v0, v1, v2)
        t1_area = ef.triangle_area(v0, v3, v1)
        e_length = ef.edge_length(v0, v1)
        rest_heights[i] = ef.height(t0_area, t1_area, e_length)

@ti.kernel
def init_rest_adj_tri_metadata(
        adj_triangles: ti.template(),
        vertices: ti.types.ndarray(dtype=ti.f32),
        meta_data: ti.template()):
    """
    Populates meta_data with the hinge edge length, the dihedral angle and the 1/3 of the average height.
    """
    for i in range(adj_triangles.shape[0]):
        v0_idx, v1_idx, v2_idx, v3_idx = adj_triangles[i]
        v0 = get_vertex_coords(v0_idx, vertices)
        v1 = get_vertex_coords(v1_idx, vertices)
        v2 = get_vertex_coords(v2_idx, vertices)
        v3 = get_vertex_coords(v3_idx, vertices)

        normal0 = ef.triangle_normal(v0, v1, v2)
        normal1 = ef.triangle_normal(v0, v3, v1)
        t0_area = ef.triangle_area(v0, v1, v2)
        t1_area = ef.triangle_area(v0, v3, v1)
        e_length = ef.edge_length(v0, v1)

        meta_data[i][0] = e_length
        meta_data[i][1] = ef.dihedral_angle(normal0, normal1)
        meta_data[i][2] = ef.height(t0_area, t1_area, e_length)

