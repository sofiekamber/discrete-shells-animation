import taichi as ti

from modules.energy_iterators import populate_edge_hessian, populate_edge_jacobian


@ti.kernel
def update_vertices(vertices:ti.types.ndarray(dtype=ti.float32, ndim=1), vertices_gui: ti.template()):
    for i in range(vertices_gui.shape[0]):
        index = i * 3
        vertices_gui[i] = ti.Vector([vertices[index], vertices[index + 1], vertices[index + 2]])

@ti.kernel
def fillInveredMass(A: ti.types.sparse_matrix_builder(), n: ti.i32):
    for i in range(n):
        A[i, i] += 1/3

@ti.func
def buildMatrix(rows: ti.i32, cols: ti.i32):
    return ti.linalg.SparseMatrixBuilder(int(rows), int(cols))

@ti.kernel
def fillIdentity(A: ti.types.sparse_matrix_builder(), n: ti.i32):
    for i in range(n):
        A[i, i] += 1  # Only +=  and -= operators are supported for now.

def newmark_integration(x:ti.types.ndarray(), delta_t:ti.f64, beta:ti.f64, e_ids: ti.template(), rest_edge_lengths: ti.template(), n_edges: ti.i32):
    """Newmark Integration:
        x_(i+1) = x_i + dt_i + dt_i^2 * ((0.5 - beta) x''_i + beta * x_''(i+1),
        x'_(i+1) = x'_i + dt_i * ((1-gamma) x''_i + gamma * x''_(i+1))"""

    n_vertices = x.shape[0]

    # create inverse mass matrix M^-1
    # M is diagonal, and the mass assigned to a vertex is a third of the total area of the incident triangles, scaled by the area mass density.
    M_inverse_builder = ti.linalg.SparseMatrixBuilder(num_rows=n_vertices, num_cols=n_vertices, max_num_triplets=100)
    fillInveredMass(M_inverse_builder, n_vertices)
    M_inverse = M_inverse_builder.build()

    # create Jacobian
    J = ti.ndarray(float, 3 * n_vertices)
    populate_edge_jacobian(J, e_ids, x, rest_edge_lengths, n_edges)

    acc_i = M_inverse @ J

    Identity_builder = ti.linalg.SparseMatrixBuilder(n_vertices, n_vertices)

    fillIdentity(Identity_builder, n_vertices)
    Identity = Identity_builder.build()

    epsilon = 1e-6
    x_old = x
    x_new = x + 1
    while x_new - x_old > epsilon:
        # create Hessian (3n x 3n)
        H_builder = ti.linalg.SparseMatrixBuilder(3 * n_vertices, 3 * n_vertices)
        populate_edge_hessian(H_builder, e_ids, x_old, rest_edge_lengths, n_edges)
        H = H_builder.build()

        A = delta_t * delta_t * beta * M_inverse * H - Identity

        b = x + delta_t + delta_t * delta_t * (beta - 0.5) * acc_i - A @ x

        solver = ti.linalg.SparseSolver(solver_type="LLT")
        solver.analyze_pattern(A)
        solver.factorize(A)

        x_old = x_new
        x_new = solver.solve(b)

    return x_new
