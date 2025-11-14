import jax
import jax.numpy as jnp


@jax.jit
def jax_compute_second_derivatives(x, y):
    """
    Compute second derivatives for cubic spline using not-a-knot boundary conditions.
    
    Parameters
    ----------
    x : array (must have length greater than 3)
        Grid points (must be strictly increasing)
    y : array
        Function values at grid points
        
    Returns
    -------
    M : array
        Second derivatives at grid points
    """
    n = len(x)
    h = jnp.diff(x)

    A = jnp.zeros((n, n))
    b = jnp.zeros(n)

    A = A.at[0, 0].set(h[1])
    A = A.at[0, 1].set(-(h[0] + h[1]))
    A = A.at[0, 2].set(h[0])
    b = b.at[0].set(0.0)

    def interior_row(i, Ab):
        A_curr, b_curr = Ab
        A_curr = A_curr.at[i, i-1].set(h[i-1])
        A_curr = A_curr.at[i, i].set(2.0 * (h[i-1] + h[i]))
        A_curr = A_curr.at[i, i+1].set(h[i])
        b_curr = b_curr.at[i].set(
            6.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        )
        return A_curr, b_curr
    
    A, b = jax.lax.fori_loop(1, n-1, interior_row, (A, b))

    A = A.at[n-1, n-3].set(h[n-2])
    A = A.at[n-1, n-2].set(-(h[n-3] + h[n-2]))
    A = A.at[n-1, n-1].set(h[n-3])
    b = b.at[n-1].set(0.0)

    M = jnp.linalg.solve(A, b)
    
    return M


@jax.jit
def jax_precompute_splines(grid, values):
    """
    Precompute second derivatives for cubic splines along each dimension.
    """
    nx, ny = values.shape

    # Compute second derivatives for splines along x (for each y)
    M_x = jnp.zeros((nx, ny))

    def compute_deriv_loop_x(i, M_x):
        M_x = M_x.at[:, i].set(
            jax_compute_second_derivatives(
                grid[0], values[:, i]
            )
        )
        return M_x
    M_x = jax.lax.fori_loop(0, ny, compute_deriv_loop_x, M_x)

    # for j in range(ny):
    #     M_x[:, j] = jax_compute_second_derivatives(
    #         grid[0], values[:, j]
    #     )


    
    # Compute second derivatives for splines along y (for each x)
    M_y = jnp.zeros((nx, ny))

    def compute_deriv_loop_y(i, M_y):
        M_y = M_y.at[i, :].set(
            jax_compute_second_derivatives(
                grid[1], values[i, :]
            )
        )
        return M_y
    M_y = jax.lax.fori_loop(0, nx, compute_deriv_loop_y, M_y)

    # for i in range(nx):
    #     M_y[i, :] = jax_compute_second_derivatives(
    #         grid[1], values[i, :]
    #     )

    return M_x, M_y



@jax.jit
def cubic_spline_evaluate(xi, grid, values, M_x, M_y, fill_value=jnp.nan):
    
    """
    Evaluate interpolator at given points using separable bicubic interpolation.
    
    This vectorized version processes all points at once for better performance.
    
    Parameters
    ----------
    xi : array-like
        Points at which to interpolate. Shape (..., 2)
        Last dimension corresponds to (x, y) coordinates.
        
    Returns
    -------
    result : array
        Interpolated values
    """
    
    original_shape = xi.shape[:-1]
    xi = xi.reshape(-1, 2)
    n_points = len(xi)
    
    x_pts = xi[:, 0]
    y_pts = xi[:, 1]
    
    # Check bounds
    out_of_bounds = jnp.zeros(n_points, dtype=bool)
    out_of_bounds |= (x_pts < grid[0][0]) | (x_pts > grid[0][-1])
    out_of_bounds |= (y_pts < grid[1][0]) | (y_pts > grid[1][-1])

    # Clamp to boundaries
    x_pts = jnp.clip(x_pts, grid[0][0], grid[0][-1])
    y_pts = jnp.clip(y_pts, grid[1][0], grid[1][-1])

    # Find x intervals for all points at once
    i_x = jnp.searchsorted(grid[0], x_pts) - 1
    i_x = jnp.clip(i_x, 0, len(grid[0]) - 2)

    # Find y intervals for all points at once
    i_y = jnp.searchsorted(grid[1], y_pts) - 1
    i_y = jnp.clip(i_y, 0, len(grid[1]) - 2)

    # Vectorized computation of normalized coordinates
    h_x = grid[0][i_x + 1] - grid[0][i_x]
    t_x = (x_pts - grid[0][i_x]) / h_x

    h_y = grid[1][i_y + 1] - grid[1][i_y]
    t_y = (y_pts - grid[1][i_y]) / h_y

    # Get corner values for all points
    z00 = values[i_x, i_y]
    z10 = values[i_x + 1, i_y]
    z01 = values[i_x, i_y + 1]
    z11 = values[i_x + 1, i_y + 1]
    
    # Get second derivatives at corners
    Mx00 = M_x[i_x, i_y]
    Mx10 = M_x[i_x + 1, i_y]
    Mx01 = M_x[i_x, i_y + 1]
    Mx11 = M_x[i_x + 1, i_y + 1]
    
    # Interpolate along x at y_j and y_{j+1} using vectorized operations
    # At y = y_j (bottom edge)
    f_x0 = (1 - t_x) * z00 + t_x * z10 + \
            ((1 - t_x)**3 - (1 - t_x)) * Mx00 * h_x**2 / 6.0 + \
            (t_x**3 - t_x) * Mx10 * h_x**2 / 6.0
    
    # At y = y_{j+1} (top edge)
    f_x1 = (1 - t_x) * z01 + t_x * z11 + \
            ((1 - t_x)**3 - (1 - t_x)) * Mx01 * h_x**2 / 6.0 + \
            (t_x**3 - t_x) * Mx11 * h_x**2 / 6.0
    
    # Get second derivatives in y-direction at the x-interpolated points
    # We need M values for the y-spline through (f_x0, f_x1)
    # For a simple 2-point case, we can use the precomputed M_y values
    # and interpolate them along x as well
    
    My00 = M_y[i_x, i_y]
    My10 = M_y[i_x + 1, i_y]
    My01 = M_y[i_x, i_y + 1]
    My11 = M_y[i_x + 1, i_y + 1]
    
    # Interpolate M_y along x at both y values
    My_x0 = (1 - t_x) * My00 + t_x * My10
    My_x1 = (1 - t_x) * My01 + t_x * My11
    
    # Now interpolate along y using the interpolated values and M's
    result = (1 - t_y) * f_x0 + t_y * f_x1 + \
                ((1 - t_y)**3 - (1 - t_y)) * My_x0 * h_y**2 / 6.0 + \
                (t_y**3 - t_y) * My_x1 * h_y**2 / 6.0
    
    # Apply fill value for out of bounds points
    # result[out_of_bounds] = fill_value
    result = jnp.where(out_of_bounds, fill_value, result)
    
    return result.reshape(original_shape)