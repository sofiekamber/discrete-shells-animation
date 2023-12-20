import math

import taichi as ti

from modules.energy_iterators import populate_edge_hessian, populate_edge_jacobian, populate_area_jacobian, \
    populate_area_hessian, populate_flex_jacobian, populate_flex_hessian
from modules.helpers import global_idx, id_x, id_y, id_z


@ti.kernel
def add_random_noise(x: ti.types.ndarray(ti.float32, 1), delta: ti.float32, n_vertices:int):
    for i in range(n_vertices):
        x[i] += ti.randn() * delta;

@ti.kernel 
def enforce_sphere_constraint(x: ti.types.ndarray(dtype=ti.float32, ndim=1), r2: ti.float32, n_vertices: int):
    n = n_vertices / 3
    for i in range(n):
        l_r = x[id_x(i)]**2 + x[id_y(i)]**2 + x[id_z(i)]**2
        if l_r > r2:
            x[id_x(i)] *= ti.sqrt(r2 / l_r)
            x[id_y(i)] *= ti.sqrt(r2 / l_r)
            x[id_z(i)] *= ti.sqrt(r2 / l_r)

@ti.kernel
def update_vertices(vertices: ti.types.ndarray(dtype=ti.float32, ndim=1), vertices_gui: ti.template()):
    for i in range(vertices_gui.shape[0]):
        index = i * 3
        vertices_gui[i] = ti.Vector([vertices[index], vertices[index + 1], vertices[index + 2]])


@ti.kernel
def fillInveredMass(A: ti.types.sparse_matrix_builder(), n: ti.i32):
    for i in range(n):
        A[i, i] += 1 / 3


@ti.func
def buildMatrix(rows: ti.i32, cols: ti.i32):
    return ti.linalg.SparseMatrixBuilder(int(rows), int(cols))


@ti.kernel
def fillIdentity(A: ti.types.sparse_matrix_builder(), n: ti.i32):
    for i in range(n):
        A[i, i] += 1  # Only +=  and -= operators are supported for now.


@ti.kernel
def add_scalar_to_ndarray(arr: ti.types.ndarray(), scalar: ti.f32):
    for i in ti.grouped(arr):
        arr[i] = arr[i] + scalar

@ti.kernel
def add_velocity_dt_to_x(x: ti.types.ndarray(), v: ti.types.ndarray(), dt: ti.f32):
    for i in range(x.shape[0]):
        x[i] = x[i] + v[i] * dt

@ti.kernel
def update_velocity_vector(v_i: ti.types.ndarray(), a_i: ti.types.ndarray(),a_i1: ti.types.ndarray(),
                             dt_i: ti.f64, gamma:ti.f64):
    """x'_(i+1) = x'_i + dt_i * ((1-gamma) x''_i + gamma * x''_(i+1))"""
    for i in range(v_i.shape[0]):
        v_i[i] = v_i[i] + dt_i * ((1 - gamma) * a_i[i] + gamma * a_i1[i])


# def unroll(arr : ti.types.ndarray(), arr_unrolled: ti.types.ndarray()):
#     for i in range(arr.shape[0]):
#         x1 = arr[i, 0]
#         y1 = arr[i, 1]
#         z1 = arr[i, 2]
#
#         # x1, y1, z1 = arr[i]
#         # x1, y1, z1 = global_idx(i)
#         arr_unrolled[3 * i] = x1
#         arr_unrolled[3 * i + 1] = y1
#         arr_unrolled[3 * i + 2] = z1
#
@ti.kernel
def roll(arr_unrolled: ti.types.ndarray(), arr: ti.types.ndarray()):
    for i in ti.grouped(arr):
        arr[i] = ti.Vector([arr_unrolled[3 * i], arr_unrolled[3 * i + 1], arr_unrolled[3 * i + 2]])


@ti.kernel
def fill_row_vector(vec: ti.types.sparse_matrix_builder(), arr: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        vec[0, i] += arr[i]


@ti.kernel
def fill_col_vector(vec: ti.types.sparse_matrix_builder(), arr: ti.types.ndarray(), n: ti.i32):
    for i in range(n):
        vec[i, 0] += arr[i]


# @ti.kernel
def fill_arr_from_col_vec(arr: ti.types.ndarray(), vec: ti.types.template(), n: ti.i32):
    for i in range(n):
        arr[i] = vec[i, 0]


# @ti.kernel
def get_euclidian_distance(x_old: ti.types.ndarray(), x_new: ti.types.ndarray(), n: ti.i32) -> ti.i32:
    distance = 0
    for i in range(n):
        distance += (x_old[i] - x_new[i]) ** 2
    return math.sqrt(distance)

@ti.kernel
def squared_norm(x: ti.types.ndarray(float), n: ti.i32) -> ti.f32:
    sum = 0.0
    for i in range(n):
        sum += x[i] ** 2
    return sum

#ti.kernel
def average(x: ti.types.ndarray(float), y: ti.types.ndarray(float), n: ti.i32) -> ti.types.ndarray(float):
    output = ti.ndarray(float, n)
    for i in range(n):
        output[i] = 0.5*x[i] + 0.5*y[i]
    return output


def newton_step(x_old: ti.types.ndarray(), x_new: ti.types.ndarray(), x_i: ti.types.ndarray(), v_i: ti.types.ndarray(),
                delta_t: ti.f64, beta: ti.f64,
                e_ids: ti.template(), rest_edge_lengths: ti.template(), n_edges: ti.i32, t_ids: ti.template(),
                A_bars: ti.template(), n_tris: ti.i32, n_vertices: ti.i32, M_inverse: ti.template(),
                Identity: ti.template(), acc_i: ti.template(), adj_t_ids: ti.template(), rest_adj_tri_metadata: ti.template(), n_adj_triangles: ti.i32, g_old: ti.f32):
    """
    Takes one newton step for finding x_i+1
    """
    x_mid = average(x_old, x_new, n_vertices)
    x_old = x_new

    H_builder = ti.linalg.SparseMatrixBuilder(n_vertices, n_vertices, 81*n_tris + 36*n_edges + 144*n_adj_triangles)
    populate_area_hessian(H_builder, t_ids, x_old, A_bars, n_tris)
    populate_edge_hessian(H_builder, e_ids, x_old, rest_edge_lengths, n_edges)
    #populate_flex_hessian(H_builder, adj_t_ids, x_old, rest_adj_tri_metadata, n_adj_triangles)
    H = H_builder.build()
    #print("H: ", H)

    J_builder = ti.linalg.SparseMatrixBuilder(n_vertices, 1, 9*n_tris + 6*n_edges + 12*n_adj_triangles)
    J_arr = ti.ndarray(ti.f32, n_vertices)
    populate_area_jacobian(J_arr, t_ids, x_old, A_bars, n_tris)
    populate_edge_jacobian(J_arr, e_ids, x_old, rest_edge_lengths, n_edges)
#    populate_flex_jacobian(J_arr, adj_t_ids, x_old, rest_adj_tri_metadata, n_adj_triangles)
    fill_col_vector(J_builder, J_arr, n_vertices)
    J = J_builder.build()
   #print("J: ", J)

    A = delta_t * delta_t * beta * M_inverse @ H + Identity

    t1 = x_i
    add_velocity_dt_to_x(t1,v_i,delta_t)
    t1_vec_builder = ti.linalg.SparseMatrixBuilder(n_vertices, 1)
    fill_col_vector(t1_vec_builder, t1, n_vertices)
    t1_vec = t1_vec_builder.build()

    #print("x_i: ", x_i.to_numpy())
    #print("v_i: ", v_i.to_numpy())
    #print("v1_vec: ", t1_vec)
    #print("adj_tri: ", rest_adj_tri_metadata.to_numpy())

    acc_i_vec_builder = ti.linalg.SparseMatrixBuilder(n_vertices, 1)
    fill_col_vector(acc_i_vec_builder, acc_i, n_vertices)
    acc_i_vec = acc_i_vec_builder.build()

    x_vec_builder = ti.linalg.SparseMatrixBuilder(n_vertices, 1)
    fill_col_vector(x_vec_builder, x_old, n_vertices)
    x_vec = x_vec_builder.build()

    #print("--------------------------------------------")
    g_arr = ti.ndarray(float, n_vertices)
    g_mat =(t1_vec + (delta_t * delta_t * (0.5 - beta))* acc_i_vec) - delta_t * delta_t*beta*M_inverse @ J - x_vec
    #print("g_mat: ", g_mat)
    fill_arr_from_col_vec(g_arr, g_mat, n_vertices)
    #print("g_arr: ", g_arr)
    g = squared_norm(g_arr, n_vertices)
    #print(g)
    if g > g_old:
        return x_old, x_mid, g

    b = (t1_vec + (delta_t * delta_t * (0.5 - beta))* acc_i_vec) - delta_t*delta_t*beta*M_inverse @ (J - (H @ x_vec))

    b_arr = ti.ndarray(float, n_vertices)
    fill_arr_from_col_vec(b_arr, b, n_vertices)

    solver = ti.linalg.SparseSolver(solver_type="LU")
    solver.analyze_pattern(A)
    solver.factorize(A)

    x_new =solver.solve(b_arr)
    
    #print((A @ x_new).to_numpy())
    #print(b_arr.to_numpy())

    return x_old, x_new, g


def newmark_integration(x_i: ti.types.ndarray(),
                        v_i: ti.types.ndarray(),
                        delta_t: ti.f64,
                        beta: ti.f64,
                        gamma: ti.f64,
                        e_ids: ti.template(),
                        rest_edge_lengths: ti.template(),
                        n_edges: ti.i32,
                        t_ids: ti.template(),
                        A_bars: ti.template(),
                        n_tris: ti.i32,
                        adj_t_ids: ti.template(),
                        rest_adj_tri_metadata: ti.template(),
                        n_adj_triangles: ti.i32):
    """Newmark Integration:
        x_(i+1) = x_i + dt_i * x'_i + dt_i^2 * ((0.5 - beta) x''_i + beta * x_''(i+1),
        x'_(i+1) = x'_i + dt_i * ((1-gamma) x''_i + gamma * x''_(i+1))"""

    n_vertices = x_i.shape[0]

    # create inverse mass matrix M^-1
    # M is diagonal, and the mass assigned to a vertex is a third of the total area of the incident triangles, scaled by the area mass density.
    # TODO: mass currently just 1/3 at diagonal
    M_inverse_builder = ti.linalg.SparseMatrixBuilder(num_rows=n_vertices, num_cols=n_vertices, max_num_triplets=100)
    fillInveredMass(M_inverse_builder, n_vertices)
    M_inverse = M_inverse_builder.build()
    
    # create Jacobian
    J = ti.ndarray(float, n_vertices)
    populate_area_jacobian(J, t_ids, x_i, A_bars, n_tris)
    populate_edge_jacobian(J, e_ids, x_i, rest_edge_lengths, n_edges)
#    populate_flex_jacobian(J, adj_t_ids, x_i, rest_adj_tri_metadata, n_adj_triangles)

    acc_i = -1.0 * M_inverse @ J

    Identity_builder = ti.linalg.SparseMatrixBuilder(n_vertices, n_vertices)

    fillIdentity(Identity_builder, n_vertices)
    Identity = Identity_builder.build()

    epsilon = 1e-6

    # first step
    # x_old = ti.ndarray(float, n_vertices)  # initial guess
    # add_scalar_to_ndarray(x_old, 1.0)
    x_old = x_i
    x_new = x_i
    # add_scalar_to_ndarray(x_new, 0.5)
    x_old, x_new, g_new = newton_step(x_old, x_new, x_i, v_i, delta_t, beta, e_ids, rest_edge_lengths, n_edges, t_ids, A_bars, n_tris,
                               n_vertices, M_inverse, Identity, acc_i, adj_t_ids, rest_adj_tri_metadata, n_adj_triangles, 1000)

    while g_new > epsilon:
        x_old, x_new, g_new = newton_step(x_old, x_new, x_i, v_i, delta_t, beta, e_ids, rest_edge_lengths, n_edges, t_ids, A_bars, n_tris,
                               n_vertices, M_inverse, Identity, acc_i, adj_t_ids, rest_adj_tri_metadata, n_adj_triangles, g_new)
        

    #print(x_new.to_numpy())

    # Update velocity
    J_new = ti.ndarray(float, n_vertices)
    populate_area_jacobian(J_new, t_ids, x_new, A_bars, n_tris)
    populate_edge_jacobian(J_new, e_ids, x_new, rest_edge_lengths, n_edges)
#    populate_flex_jacobian(J_new, adj_t_ids, x_new, rest_adj_tri_metadata, n_adj_triangles)

    acc_new = -1.0 * M_inverse @ J_new
    update_velocity_vector(v_i,acc_i,acc_new,delta_t,gamma)

    return x_new
