import taichi as ti


@ti.kernel
def newmark_integration():
    """Newmark Integration:
        x_(i+1) = x_i + dt_i + dt_i^2 * ((0.5 - beta) x''_i + beta * x_''(i+1),
        x'_(i+1) = x'_i + dt_i * ((1-gamma) x''_i + gamma * x''_(i+1))"""
    pass