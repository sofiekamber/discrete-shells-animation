import taichi as ti

@ti.kernel
def update_vertices(vertices: ti.types.ndarray(dtype=ti.math.vec3, ndim=1), vertices_gui: ti.template()):
    for i in range(vertices_gui.shape[0]):
        vertices_gui[i] = vertices[i]

@ti.kernel
def fillIdentity(A: ti.types.sparse_matrix_builder(), n: ti.i32):
    for i in range(n):
        A[i, i] += 1  # Only +=  and -= operators are supported for now.

@ti.kernel
def newmark_integration(x:ti.template(), delta_t:ti.f64, beta:ti.f64):
    """Newmark Integration:
        x_(i+1) = x_i + dt_i + dt_i^2 * ((0.5 - beta) x''_i + beta * x_''(i+1),
        x'_(i+1) = x'_i + dt_i * ((1-gamma) x''_i + gamma * x''_(i+1))"""

    n_vertices = x.shape[0]

    # create inverse mass matrix M^-1
    M_inverse = ti.linalg.SparseMatrixBuilder(n_vertices, n_vertices) # TODO: replace with actual inverted Mass matrix

    # create Jacobian
    J = ti.linalg.SparseMatrixBuilder(n_vertices, 3) # TODO: replace with actual Jacobian

    acc_i = M_inverse @ J

    Identity = ti.linalg.SparseMatrixBuilder(n_vertices, n_vertices)

    fillIdentity(Identity, n_vertices)

    epsilon = 1e-6
    x_old = x
    x_new = x + 1
    while x_new - x_old > epsilon:
        # create Hessian
        H = ti.linalg.SparseMatrixBuilder(n_vertices, 3 * 3)  # TODO: replace with actual Hessian

        A = delta_t * delta_t * beta * M_inverse * H - Identity.build()

        b = x + delta_t + delta_t * delta_t * (beta - 0.5) * acc_i - A @ x

        solver = ti.linalg.SparseSolver(solver_type="LLT")
        solver.analyze_pattern(A)
        solver.factorize(A)

        x_old = x_new
        x_new = solver.solve(b)

    return x_new
