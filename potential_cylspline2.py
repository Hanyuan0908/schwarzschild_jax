import numpy as np
from tqdm import tqdm
from typing import Callable, Optional, Tuple, Dict, Union, List
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gamma, hyp2f1
from constants import *
import jax


Array = np.ndarray
ArrayLike = Union[float, np.ndarray]

# import jax.numpy as jnp
# # import numpy as np

class CubicSpline2D:
    """
    2D cubic spline interpolator for regular grids.
    
    This class mimics scipy.interpolate.RegularGridInterpolator with method='cubic'
    for 2D data only. It uses separable bicubic spline interpolation.
    
    Parameters
    ----------
    points : tuple of 2 arrays
        Grid points in each dimension. Each array must be strictly increasing.
        points = (x_grid, y_grid)
    values : 2D array
        Data values at grid points. Shape must be (len(x_grid), len(y_grid))
    bounds_error : bool, optional
        If True, raises ValueError for out-of-bounds points. Default is True.
    fill_value : float, optional
        Value to return for out-of-bounds points if bounds_error=False.
        Default is np.nan.
    """
    
    def __init__(self, points, values, bounds_error=True, fill_value=np.nan):
        if len(points) != 2:
            raise ValueError("This implementation only supports 2D interpolation")
        
        self.grid = tuple(np.asarray(p, dtype=float) for p in points)
        self.values = np.asarray(values, dtype=float)
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        
        # Validate inputs
        if self.values.shape != tuple(len(g) for g in self.grid):
            raise ValueError(f"values shape {self.values.shape} doesn't match "
                           f"grid shape {tuple(len(g) for g in self.grid)}")
        
        for i, g in enumerate(self.grid):
            if len(g) < 2:
                raise ValueError(f"Grid dimension {i} must have at least 2 points")
            if not np.all(np.diff(g) > 0):
                raise ValueError(f"Grid dimension {i} must be strictly increasing")
        
        # Precompute spline second derivatives
        self._precompute_splines()
    
    def _precompute_splines(self):
        """
        Precompute second derivatives for cubic splines along each dimension.
        """
        nx, ny = self.values.shape
        
        # Compute second derivatives for splines along x (for each y)
        self.M_x = np.zeros((nx, ny))
        for j in range(ny):
            self.M_x[:, j] = self._compute_second_derivatives(
                self.grid[0], self.values[:, j]
            )
        
        # Compute second derivatives for splines along y (for each x)
        self.M_y = np.zeros((nx, ny))
        for i in range(nx):
            self.M_y[i, :] = self._compute_second_derivatives(
                self.grid[1], self.values[i, :]
            )
    
    def _compute_second_derivatives(self, x, y):
        """
        Compute second derivatives for cubic spline using not-a-knot boundary conditions.
        
        Parameters
        ----------
        x : array
            Grid points (must be strictly increasing)
        y : array
            Function values at grid points
            
        Returns
        -------
        M : array
            Second derivatives at grid points
        """
        n = len(x)
        h = np.diff(x)
        
        if n == 2:
            # Linear interpolation for 2 points
            return np.zeros(2)
        
        # Build tridiagonal system
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        # First equation: not-a-knot at left
        if n == 3:
            A[0, 0] = 1.0
            b[0] = 0.0
        else:
            A[0, 0] = h[1]
            A[0, 1] = -(h[0] + h[1])
            A[0, 2] = h[0]
            b[0] = 0.0
        
        # Interior equations
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2.0 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 6.0 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        
        # Last equation: not-a-knot at right
        if n == 3:
            A[n-1, n-1] = 1.0
            b[n-1] = 0.0
        else:
            A[n-1, n-3] = h[n-2]
            A[n-1, n-2] = -(h[n-3] + h[n-2])
            A[n-1, n-1] = h[n-3]
            b[n-1] = 0.0
        
        # Solve the system
        M = np.linalg.solve(A, b)
        
        return M
    
    def __call__(self, xi, method=None):
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
        xi = np.asarray(xi, dtype=float)
        
        # Handle shape
        if xi.shape[-1] != 2:
            raise ValueError("Last dimension of xi must be 2 for 2D interpolation")
        
        original_shape = xi.shape[:-1]
        xi = xi.reshape(-1, 2)
        n_points = len(xi)
        
        x_pts = xi[:, 0]
        y_pts = xi[:, 1]
        
        # Check bounds
        out_of_bounds = np.zeros(n_points, dtype=bool)
        out_of_bounds |= (x_pts < self.grid[0][0]) | (x_pts > self.grid[0][-1])
        out_of_bounds |= (y_pts < self.grid[1][0]) | (y_pts > self.grid[1][-1])
        
        if self.bounds_error and np.any(out_of_bounds):
            raise ValueError("Some points are out of bounds")
        
        # Clamp to boundaries
        x_pts = np.clip(x_pts, self.grid[0][0], self.grid[0][-1])
        y_pts = np.clip(y_pts, self.grid[1][0], self.grid[1][-1])
        
        # Find x intervals for all points at once
        i_x = np.searchsorted(self.grid[0], x_pts) - 1
        i_x = np.clip(i_x, 0, len(self.grid[0]) - 2)
        
        # Find y intervals for all points at once
        i_y = np.searchsorted(self.grid[1], y_pts) - 1
        i_y = np.clip(i_y, 0, len(self.grid[1]) - 2)
        
        # Vectorized computation of normalized coordinates
        h_x = self.grid[0][i_x + 1] - self.grid[0][i_x]
        t_x = (x_pts - self.grid[0][i_x]) / h_x
        
        h_y = self.grid[1][i_y + 1] - self.grid[1][i_y]
        t_y = (y_pts - self.grid[1][i_y]) / h_y
        
        # Get corner values for all points
        z00 = self.values[i_x, i_y]
        z10 = self.values[i_x + 1, i_y]
        z01 = self.values[i_x, i_y + 1]
        z11 = self.values[i_x + 1, i_y + 1]
        
        # Get second derivatives at corners
        Mx00 = self.M_x[i_x, i_y]
        Mx10 = self.M_x[i_x + 1, i_y]
        Mx01 = self.M_x[i_x, i_y + 1]
        Mx11 = self.M_x[i_x + 1, i_y + 1]
        
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
        
        My00 = self.M_y[i_x, i_y]
        My10 = self.M_y[i_x + 1, i_y]
        My01 = self.M_y[i_x, i_y + 1]
        My11 = self.M_y[i_x + 1, i_y + 1]
        
        # Interpolate M_y along x at both y values
        My_x0 = (1 - t_x) * My00 + t_x * My10
        My_x1 = (1 - t_x) * My01 + t_x * My11
        
        # Now interpolate along y using the interpolated values and M's
        result = (1 - t_y) * f_x0 + t_y * f_x1 + \
                 ((1 - t_y)**3 - (1 - t_y)) * My_x0 * h_y**2 / 6.0 + \
                 (t_y**3 - t_y) * My_x1 * h_y**2 / 6.0
        
        # Apply fill value for out of bounds points
        result[out_of_bounds] = self.fill_value
        
        return result.reshape(original_shape)

# -------------------- coordinate helpers --------------------

def cartesian_to_cylindrical(x: Array, y: Array, z: Array) -> Tuple[Array, Array, Array]:
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    R  = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0.0, phi + 2*np.pi, phi)
    phi = np.where(R == 0.0, 0.0, phi)  # define φ=0 on axis
    return R, phi, z

def cylindrical_to_cartesian(R: Array, phi: Array, z: Array) -> Tuple[Array, Array, Array]:
    R = np.asarray(R, float); phi = np.asarray(phi, float); z = np.asarray(z, float)
    x = R * np.cos(phi); y = R * np.sin(phi)
    return x, y, z



# def _hypergeom_m(m: int, x: float, want_deriv: bool = False):
#     if x < X_THRESHOLD1[m]:
#         # print('hallo')
#         A = HYPERGEOM_0[m] if x < X_THRESHOLD0[m] else HYPERGEOM_I[m]
#         xA8 = x + A[8]
#         xA6 = x + A[6] + A[7] / xA8
#         xA4 = x + A[4] + A[5] / xA6
#         xA2 = x + A[2] + A[3] / xA4
#         F = A[0] + A[1] / xA2
#         if want_deriv:
#             dFdx = -A[1] / (xA2**2) * (1 - A[3] / (xA4**2) * (1 - A[5] / (xA6**2) * (1 - A[7] / (xA8**2))))
#         else:
#             dFdx = 0.0
#         return F, dFdx
#     else:
#         A = HYPERGEOM_1[m]
#         y = 1.0 - x
#         y2 = y*y
#         z = np.log(max(y, np.finfo(float).tiny))
#         F = (A[0] + A[1]*z +
#              (A[2] + A[3]*z) * y +
#              (A[4] + A[5]*z + (A[6] + A[7]*z) * y + (A[8] + A[9]*z) * y2) * y2)
#         if want_deriv:
#             dFdx = (-A[1]/y
#                     - (A[2] + A[3] + A[3]*z)
#                     - (2*A[4] + A[5] + 2*A[5]*z) * y
#                     - (3*A[6] + A[7] + 3*A[7]*z + (4*A[8] + A[9] + 4*A[9]*z) * y) * y2)
#         else:
#             dFdx = 0.0
#         return F, dFdx

# def legendreQ(n: float, x: ArrayLike) -> np.ndarray:
#     xa = np.asarray(x, float)
#     xa = np.maximum(xa, 1.0)
#     out = np.empty_like(xa)
#     m = int(round(n + 0.5))
#     is_halfint = (abs(m - (n + 0.5)) < 1e-12) and (0 <= m <= MMAX_HYPERGEOM)
#     # it = np.nditer(xa, flags=['multi_index'])
#     # while not it.finished:
#     #     xv = float(it[0])
#     #     if is_halfint:
#     #         pref = Q_PREFACTOR[m] / np.sqrt(xv) / (xv**m)
#     #         F, _ = _hypergeom_m(m, 1.0/(xv*xv), want_deriv=False)
#     #         out[it.multi_index] = pref * F
#     #     else:
#     #         C = (2.0 * xv) ** (-1.0 - n) * np.sqrt(np.pi) * gamma(n + 1.0) / gamma(n + 1.5)
#     #         out[it.multi_index] = C * hyp2f1(1.0 + n/2.0, 0.5 + n/2.0, 1.5 + n, 1.0/(xv*xv))
#     #     it.iternext()
#     for idx, xv in enumerate(xa):
#         if is_halfint:
#             pref = Q_PREFACTOR[m] / np.sqrt(xv) / (xv**m)
#             F, _ = _hypergeom_m(m, 1.0/(xv*xv), want_deriv=False)
#             out[idx] = pref * F
#         else:
#             C = (2.0 * xv) ** (-1.0 - n) * np.sqrt(np.pi) * gamma(n + 1.0) / gamma(n + 1.5)
#             out[idx] = C * hyp2f1(1.0 + n/2.0, 0.5 + n/2.0, 1.5 + n, 1.0/(xv*xv))
#     return out

def _hypergeom_m(m: int, x: ArrayLike):
    # if x < X_THRESHOLD1[m]:
    #     # print('hallo')
    #     A = HYPERGEOM_0[m] if x < X_THRESHOLD0[m] else HYPERGEOM_I[m]
    #     xA8 = x + A[8]
    #     xA6 = x + A[6] + A[7] / xA8
    #     xA4 = x + A[4] + A[5] / xA6
    #     xA2 = x + A[2] + A[3] / xA4
    #     F = A[0] + A[1] / xA2

    #     dFdx = 0.0

    # else:
    #     A = HYPERGEOM_1[m]
    #     y = 1.0 - x
    #     y2 = y*y
    #     z = np.log(max(y, np.finfo(float).tiny))
    #     F = (A[0] + A[1]*z +
    #          (A[2] + A[3]*z) * y +
    #          (A[4] + A[5]*z + (A[6] + A[7]*z) * y + (A[8] + A[9]*z) * y2) * y2)

    #     dFdx = 0.0


    y = 1.0 - x
    y2 = y*y
    # z = np.log(np.amax(y, 1e-12))
    z = np.log(np.where(y > 1e-12, y, 1e-12))

    HYPERGEOM_0_m = HYPERGEOM_0[m]
    HYPERGEOM_I_m = HYPERGEOM_I[m]
    HYPERGEOM_1_m = HYPERGEOM_1[m]

    xA8_1 = x + HYPERGEOM_0_m[8]
    xA6_1 = x + HYPERGEOM_0_m[6] + HYPERGEOM_0_m[7] / xA8_1
    xA4_1 = x + HYPERGEOM_0_m[4] + HYPERGEOM_0_m[5] / xA6_1
    xA2_1 = x + HYPERGEOM_0_m[2] + HYPERGEOM_0_m[3] / xA4_1
    val_1 = HYPERGEOM_0_m[0] + HYPERGEOM_0_m[1] / xA2_1

    xA8_2 = x + HYPERGEOM_I_m[8]
    xA6_2 = x + HYPERGEOM_I_m[6] + HYPERGEOM_I_m[7] / xA8_2
    xA4_2 = x + HYPERGEOM_I_m[4] + HYPERGEOM_I_m[5] / xA6_2
    xA2_2 = x + HYPERGEOM_I_m[2] + HYPERGEOM_I_m[3] / xA4_2
    val_2 = HYPERGEOM_I_m[0] + HYPERGEOM_I_m[1] / xA2_2

    val3 = (HYPERGEOM_1_m[0] + HYPERGEOM_1_m[1]*z +
             (HYPERGEOM_1_m[2] + HYPERGEOM_1_m[3]*z) * y +
             (HYPERGEOM_1_m[4] + HYPERGEOM_1_m[5]*z + 
             (HYPERGEOM_1_m[6] + HYPERGEOM_1_m[7]*z) * y + 
             (HYPERGEOM_1_m[8] + HYPERGEOM_1_m[9]*z) * y2) * y2)

    F = np.where(x < X_THRESHOLD1[m],
                 np.where(x < X_THRESHOLD0[m], val_1, val_2),
                 val3)

    return F

def legendreQ(n: float, x: ArrayLike) -> np.ndarray:
    xa = np.asarray(x, float)
    xa = np.maximum(xa, 1.0)
    out = np.empty_like(xa)
    m = int(round(n + 0.5))
    is_halfint = (abs(m - (n + 0.5)) < 1e-12) and (0 <= m <= MMAX_HYPERGEOM)

    # for idx, xv in enumerate(xa):
    # if is_halfint:
    #     pref = Q_PREFACTOR[m] / np.sqrt(xv) / (xv**m)
    #     F, _ = _hypergeom_m(m, 1.0/(xv*xv), want_deriv=False)
    #     out[idx] = pref * F
    # else:
    #     C = (2.0 * xv) ** (-1.0 - n) * np.sqrt(np.pi) * gamma(n + 1.0) / gamma(n + 1.5)
    #     out[idx] = C * hyp2f1(1.0 + n/2.0, 0.5 + n/2.0, 1.5 + n, 1.0/(xv*xv))

    if is_halfint:
        pref = Q_PREFACTOR[m] / np.sqrt(xa) / (xa**m)
        F = _hypergeom_m(m, 1.0/(xa*xa))
        out = pref * F
    else:
        C = (2.0 * xa) ** (-1.0 - n) * np.sqrt(np.pi) * gamma(n + 1.0) / gamma(n + 1.5)
        out = C * hyp2f1(1.0 + n/2.0, 0.5 + n/2.0, 1.5 + n, 1.0/(xa*xa))

    return out



# -------------------- CylSpline --------------------

class CylSpline:
    """
    CylSpline (z-even): build z≥0 (includes z=0), project ρ→ρ_m, compute Φ_m via 2D integration,
    fit linear 2D interpolators for both ρ_m and Φ_m, and evaluate Φ(R,φ,z).
    """

    def __init__(
        self,
        rho_xyz: Callable[[Array, Array, Array], Array],
        Rmin: float = 1e-3, Rmax: float = 20.0,
        Zmin: float = 0.0,  Zmax: float = 10.0,
        NR: int = 41, Nz_pos: int = 20,  # positive z nodes; total z nodes = Nz_pos + 1 (including 0)
        mmax: int = 8,
        *,
        R0: Optional[float] = None, nphi_quad: int = 200, G: float = 4.3e-6,
        z_spacing: str = "geometric", assume_z_even: bool = True,
    ):
        # store knobs
        self.rho_xyz = rho_xyz
        self.Rmin, self.Rmax = float(Rmin), float(Rmax)
        self.Zmin, self.Zmax = float(Zmin), float(Zmax)
        self.NR, self.Nz_pos = int(NR), int(Nz_pos)
        self.mmax = int(mmax)
        self.R0_user, self.nphi_quad, self.G = R0, int(nphi_quad), float(G)
        self.z_spacing, self.assume_z_even = str(z_spacing), bool(assume_z_even)

        # validate
        if not (self.Rmin >= 0 and self.Rmax > self.Rmin): raise ValueError("0 ≤ Rmin < Rmax")
        if not (self.Zmax > self.Zmin and self.Zmin >= 0): raise ValueError("0 ≤ Zmin < Zmax")
        if self.NR < 3 or self.Nz_pos < 1: raise ValueError("NR ≥ 3, Nz_pos ≥ 1")
        if self.mmax < 0: raise ValueError("mmax ≥ 0")
        if self.nphi_quad < 4: raise ValueError("nphi_quad ≥ 4")
        if not self.assume_z_even: raise ValueError("z≥0 build assumes even symmetry")

        # grids
        self.R = np.geomspace(max(self.Rmin, 1e-3), self.Rmax, self.NR)
        if self.z_spacing == "geometric":
            self.Zpos = np.geomspace(max(self.Zmin, 1e-3), self.Zmax, self.Nz_pos)
        elif self.z_spacing == "linear":
            self.Zpos = np.linspace(self.Zmin, self.Zmax, self.Nz_pos)
        else:
            raise ValueError("z_spacing must be 'geometric' or 'linear'")
        self.Z_nonneg = np.concatenate(([0.0], self.Zpos))

        # transform scaling (used only if you later choose to spline in transformed coords)
        self.R0_eff = float(self.R[self.NR // 2]) if self.R0_user is None else float(self.R0_user)

        # smoke test density
        Xp = np.array([max(self.Rmin, 1e-6), 0.5*(self.Rmin+self.Rmax), self.Rmax])
        Yp = np.array([0.0, 1.0, -1.0])
        Zp = np.array([0.0, max(self.Zmin, 1e-6), self.Zmax])
        X, Y, Z = np.meshgrid(Xp, Yp, Zp, indexing="ij")
        if not np.isfinite(np.asarray(self.rho_xyz(X,Y,Z))).all():
            raise ValueError("rho_xyz returned non-finite values in a basic probe.")

        # storage
        self._rho_m_grid = {}
        self._rho_m_real_interp = {}
        self._rho_m_imag_interp = {}
        self._Phi_m_grid = {}
        self._Phi_m_real_interp = {}
        self._Phi_m_imag_interp = {}

    # --------- adapters ---------
    def rho_Rzphi(self, R: Array, z: Array, phi: Array) -> Array:
        R = np.asarray(R, float)
        z = np.abs(np.asarray(z, float))  # even symmetry
        phi = np.asarray(phi, float)
        x, y, zz = cylindrical_to_cartesian(R, phi, z)
        return self.rho_xyz(x, y, zz)

    # --------- Step 2: ρ → ρ_m on the (R, z≥0) grid + linear 2D interpolators ---------
    def compute_rho_m(self) -> None:
        NR, NZ = self.NR, self.Z_nonneg.size
        Rg, Zg = np.meshgrid(self.R, self.Z_nonneg, indexing="ij")
        nphi = int(self.nphi_quad)
        phi = np.linspace(0.0, 2*np.pi, nphi, endpoint=False)
        dphi = (2*np.pi) / nphi
        rho_m = {m: np.zeros((NR, NZ), dtype=complex) for m in range(0, self.mmax+1)}

        for ph in phi:
            vals = self.rho_Rzphi(Rg, Zg, ph)     # (NR,NZ)
            exp_ph = np.exp(-1j * np.arange(self.mmax + 1) * ph)[:, None, None]
            rho_m_stack = vals[None, :, :] * exp_ph * dphi / (2.0 * np.pi)  # (mmax+1, NR, NZ)
            for m in range(0, self.mmax + 1):
                rho_m[m] += rho_m_stack[m]

        for m in range(0, self.mmax + 1):
            self._rho_m_grid[m] = rho_m[m]
            # self._rho_m_interp[m] = RegularGridInterpolator(
            #     (self.R, self.Z_nonneg), rho_m[m], method="linear",
            #     bounds_error=False, fill_value=None
            # )
            self._rho_m_real_interp[m] = CubicSpline2D(
                (self.R, self.Z_nonneg), rho_m[m].real,
                bounds_error=False, fill_value=0.
            )
            self._rho_m_imag_interp[m] = CubicSpline2D(
                (self.R, self.Z_nonneg), rho_m[m].imag,
                bounds_error=False, fill_value=0.
            )


    def rho_m_eval(self, m: int, R: ArrayLike, z: ArrayLike) -> Array:
        if m < 0 or m > self.mmax: raise ValueError(f"m∈[0,{self.mmax}]")
        if m not in self._rho_m_real_interp: raise RuntimeError("compute_rho_m() first.")
        Rb = np.asarray(R, float)
        zb = np.abs(np.asarray(z, float))
        shape = Rb.shape
        # pts = np.stack(np.broadcast_arrays(Rb, zb), axis=-1).reshape(-1, 2)
        pts = np.column_stack((Rb.ravel(), zb.ravel()))
        # print('hallo')
        # print(np.broadcast_arrays(Rb, zb))
        return self._rho_m_real_interp[m](pts).reshape(shape) + 1j * self._rho_m_imag_interp[m](pts).reshape(shape)

    # --------- kernel Ξ_m via AGAMA LegendreQ ---------
    def kernel_Xi_m(self, m: int, R: float, z: float, Rp: ArrayLike, zp: ArrayLike) -> np.ndarray:
        Rp = np.asarray(Rp, float); zp = np.asarray(zp, float)
        out = np.empty_like(Rp, dtype=float)
        if R < 1e-3:
            out[:] = 0.0 if m > 0 else 1.0 / np.sqrt(R*R + Rp*Rp + (z - zp)**2)
            return out
        mask_axis = (Rp < 1e-3)
        if np.any(mask_axis):
            out[mask_axis] = 0.0 if m > 0 else 1.0 / np.sqrt(R*R + Rp[mask_axis]**2 + (z - zp[mask_axis])**2)
        mask = ~mask_axis
        if np.any(mask):
            Rp_reg = Rp[mask]; dz = (z - zp[mask])
            chi = (R*R + Rp_reg*Rp_reg + dz*dz) / (2.0 * R * Rp_reg)
            chi = np.maximum(chi, 1.0)
            Q = legendreQ(m - 0.5, chi)
            out[mask] = (1.0 / (np.pi * np.sqrt(R * Rp_reg))) * Q
        return out

    # --------- common integrand for Φ_m ---------
    def _integrand_m(self, m: int, R: float, z: float):
        def f(Rp, zp):
            rho = self.rho_m_eval(m, Rp, zp)                 # complex
            Xi  = self.kernel_Xi_m(m, R, z, Rp, zp)          # real
            return (2.0 * np.pi) * Rp * rho * Xi
        return f

    # --------- 1D Simpson weights for tensor rule ---------
    @staticmethod
    def _simpson_weights(n: int) -> np.ndarray:
        if n < 3 or n % 2 == 0:
            raise ValueError("Simpson requires n odd and ≥3.")
        w = np.ones(n); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
        return w

    # # ===================== ADAPTIVE integrator (with eval budget) =====================
    # def _adaptive_simpson_2d(self, f, ax, bx, ay, by, tol=1e-4, max_evals=10_000):
    #     cache = {}; ncall = 0
    #     def val(x,y):
    #         nonlocal ncall
    #         k=(float(x),float(y)); v=cache.get(k)
    #         if v is None: v=f(x,y); cache[k]=v; ncall+=1
    #         return v
    #     def Srect(x0,x1,y0,y1):
    #         xm=0.5*(x0+x1); ym=0.5*(y0+y1)
    #         f00=val(x0,y0); f10=val(xm,y0); f20=val(x1,y0)
    #         f01=val(x0,ym); f11=val(xm,ym); f21=val(x1,ym)
    #         f02=val(x0,y1); f12=val(xm,y1); f22=val(x1,y1)
    #         hx=0.5*(x1-x0); hy=0.5*(y1-y0)
    #         return (hx*hy)/9.0*((f00+f20+f02+f22)+4*(f10+f01+f21+f12)+16*f11)
    #     def rec(x0,x1,y0,y1,S,tol_loc):
    #         xm=0.5*(x0+x1); ym=0.5*(y0+y1)
    #         S1=Srect(x0,xm,y0,ym); S2=Srect(xm,x1,y0,ym)
    #         S3=Srect(x0,xm,ym,y1); S4=Srect(xm,x1,ym,y1)
    #         Ssum=S1+S2+S3+S4; err=np.abs(Ssum-S)/15.0
    #         if np.all(err<=tol_loc) or ncall>=max_evals: return Ssum, err
    #         tol_child=tol_loc/4.0
    #         I1,e1=rec(x0,xm,y0,ym,S1,tol_child)
    #         if ncall>=max_evals: return I1+S2+S3+S4, e1
    #         I2,e2=rec(xm,x1,y0,ym,S2,tol_child)
    #         if ncall>=max_evals: return I1+I2+S3+S4, e2
    #         I3,e3=rec(x0,xm,ym,y1,S3,tol_child)
    #         if ncall>=max_evals: return I1+I2+I3+S4, e3
    #         I4,e4=rec(xm,x1,ym,y1,S4,tol_child)
    #         return I1+I2+I3+I4, max(e1,e2,e3,e4)
    #     S0=Srect(ax,bx,ay,by)
    #     I,err=rec(ax,bx,ay,by,S0,tol)
    #     return I, ncall, (ncall>=max_evals)

    # def compute_phi_m_grid_adaptive(
    #     self, *, tol: float = 1e-4, max_evals: int = 10_000,
    #     R_int_max: Optional[float] = None, Z_int_max: Optional[float] = None,
    #     m_list: Optional[List[int]] = None, progress: bool = False
    # ):
    #     if not self._rho_m_interp: raise RuntimeError("compute_rho_m() first.")
    #     Rint=float(R_int_max if R_int_max is not None else self.Rmax)
    #     Zint=float(Z_int_max if Z_int_max is not None else self.Zmax)
    #     m_list = list(range(self.mmax+1)) if m_list is None else list(m_list)
    #     self._Phi_m_grid={}; self._Phi_m_interp={}
    #     diagnostics={}
    #     for m in m_list:
    #         Phi=np.zeros((self.NR, self.Z_nonneg.size), dtype=complex)
    #         for i,R in enumerate(self.R):
    #             for j,z in enumerate(self.Z_nonneg):
    #                 f=self._integrand_m(m,R,z)
    #                 fre=lambda x,y: np.real(f(x,y))
    #                 fim=lambda x,y: np.imag(f(x,y))
    #                 Ire,n1,_=self._adaptive_simpson_2d(fre,0.0,Rint,0.0,Zint,tol=tol/2,max_evals=max_evals)
    #                 Iim,n2,_=self._adaptive_simpson_2d(fim,0.0,Rint,0.0,Zint,tol=tol/2,max_evals=max(0,max_evals-n1))
    #                 Phi[i,j] = -self.G * 2.0 * (Ire + 1j*Iim)  # even-in-z ⇒ ×2
    #                 diagnostics[(m,i,j)]={"evals":n1+n2}
    #             if progress: print(f"[adaptive] m={m} R[{i+1}/{self.NR}]")
    #         self._Phi_m_grid[m]=Phi
    #         self._Phi_m_interp[m]=RegularGridInterpolator(
    #             (self.R, self.Z_nonneg), Phi, method="cubic", bounds_error=False, fill_value=None
    #         )
    #     return diagnostics

    # # ===================== FIXED integrator (uniform tensor grid, N_int≈10k) =====================
    def compute_phi_m_grid_fixed(
        self, *, N_int: int = 10_000,
        R_int_max: Optional[float] = None, Z_int_max: Optional[float] = None,
        rule: str = "simpson", m_list: Optional[List[int]] = None, progress: bool = False
    ):
        if not self._rho_m_interp: raise RuntimeError("compute_rho_m() first.")
        Rint=float(R_int_max if R_int_max is not None else self.Rmax)
        Zint=float(Z_int_max if Z_int_max is not None else self.Zmax)
        m_list = list(range(self.mmax+1)) if m_list is None else list(m_list)

        nx=int(np.sqrt(max(9,N_int/1))); nz=int(max(3, (N_int* np.sqrt(1))//nx))
        # print(nx * nz)
        if rule=="simpson":
            if nx%2==0: nx+=1
            if nz%2==0: nz+=1
        Rp = np.linspace(0.0, Rint, nx);  zp = np.linspace(0.0, Zint, nz)
        dR = Rp[1]-Rp[0]; dz = zp[1]-zp[0]
        if rule=="simpson":
            wR=self._simpson_weights(nx)/3; wz=self._simpson_weights(nz)/3
            wz = wz.copy()
            # wz[0] *= 2.0 
        elif rule=="trapezoid":
            wR=np.ones(nx); wR[0]=wR[-1]=0.5; wz=np.ones(nz); wz[0]=wz[-1]=0.5
        else:
            raise ValueError("rule must be 'simpson' or 'trapezoid'")
        RpM,zpM=np.meshgrid(Rp,zp,indexing="ij"); w2d=(wR[:,None]*wz[None,:])*(dR*dz)

        self._Phi_m_grid={}; self._Phi_m_interp={}
        for m in tqdm(m_list, total = len(m_list)):
            Phi=np.zeros((self.NR, self.Z_nonneg.size), dtype=complex)
            for i,R in (enumerate(self.R)):
                for j,z in (enumerate(self.Z_nonneg)):
                    rho = self.rho_m_eval(m, RpM, zpM)               # (nx,nz) complex
                    Xi  = self.kernel_Xi_m(m, R, z, RpM, zpM)        # (nx,nz) real
                    I   = np.sum((2.0*np.pi)*RpM*rho*Xi * w2d)       # tensor quadrature
                    Phi[i,j] = -self.G * 2.0 * I
                if progress: print(f"[fixed] m={m} R[{i+1}/{self.NR}]")
            self._Phi_m_grid[m]=Phi
            # self._Phi_m_interp[m]=RegularGridInterpolator(
            #     (self.R, self.Z_nonneg), Phi, method="linear", bounds_error=False, fill_value=None
            # )
            self._Phi_m_real_interp[m]=CubicSpline2D(
                (self.R, self.Z_nonneg), Phi.real,
                bounds_error=False, fill_value=0.
            )
            self._Phi_m_imag_interp[m]=CubicSpline2D(
                (self.R, self.Z_nonneg), Phi.imag,
                bounds_error=False, fill_value=0.
            )
        return {"nx":nx,"nz":nz,"N_total":nx*nz,"rule":rule}

    def _xieta_to_Rz_jacobian(
        self,
        xi: np.ndarray,
        eta: np.ndarray,
        *,
        Rmin_map: Optional[float] = None,
        Rmax_map: Optional[float] = None,
        zmin_map: Optional[float] = None,
        zmax_map: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        AGAMA-style mapping from (xi, eta) in [0,1]^2 to (R', z') in [0,Rmax]×[0,Zmax]:

            R'(xi) = Rmin * ((1 + Rmax/Rmin)^xi - 1)
            z'(eta)= zmin * ((1 + zmax/zmin)^eta - 1)

        and the Jacobian for the change of variables (including dR/dxi and dz/deta,
        but NOT the 2πR' factor):

            dR/dxi  = ln(1 + Rmax/Rmin) * (Rmin + R')
            dz/deta = ln(1 + zmax/zmin) * (zmin + z')

        We return:
            R', z', J = (dR/dxi) * (dz/deta)

        Notes
        -----
        - For stability, we choose the inner "scale" radii like AGAMA:
            Rmin_map ≈ size of the first radial cell, i.e. self.R[1]
            zmin_map ≈ first positive z node, i.e. self.Z_nonneg[1]
        If these are not available, fall back to self.Rmin / self.Zmin.
        - The 2π R' factor is multiplied in the integrand, not here.
        """
        xi  = np.asarray(xi,  dtype=float)
        eta = np.asarray(eta, dtype=float)

        # Pick AGAMA-like inner scales if not explicitly provided
        if Rmin_map is None:
            Rmin_map = float(self.R[1]) if self.NR >= 2 else max(self.Rmin, 1e-3)
        if Rmax_map is None:
            Rmax_map = float(self.Rmax)
        if zmin_map is None:
            if self.Z_nonneg.size >= 2:
                zmin_map = float(self.Z_nonneg[1])
            else:
                zmin_map = max(self.Zmin, 1e-3)
        if zmax_map is None:
            zmax_map = float(self.Zmax)

        # Guardrails
        if not (Rmax_map > Rmin_map > 0):
            raise ValueError("Require 0 < Rmin_map < Rmax_map.")
        if not (zmax_map > zmin_map > 0):
            raise ValueError("Require 0 < zmin_map < zmax_map.")

        # Precompute logs
        LR = np.log(1.0 + Rmax_map / Rmin_map)
        LZ = np.log(1.0 + zmax_map / zmin_map)

        # Map to physical coordinates
        pR = np.power(1.0 + Rmax_map / Rmin_map, xi)
        pZ = np.power(1.0 + zmax_map / zmin_map, eta)
        Rp = Rmin_map * (pR - 1.0)
        zp = zmin_map * (pZ - 1.0)

        # Jacobian part from the coordinate transform (no 2πR' here)
        dR_dxi  = LR * (Rmin_map + Rp)
        dz_deta = LZ * (zmin_map + zp)
        J = dR_dxi * dz_deta
        return Rp, zp, J
    
    # def compute_phi_m_grid_fixed_mapped(
    #     self,
    #     *,
    #     N_int: int = 10_000,
    #     n_xi: Optional[int] = None,
    #     n_eta: Optional[int] = None,
    #     m_list: Optional[List[int]] = None,
    #     progress: bool = False,
    # ):
    #     """
    #     Compute Φ_m(R,z) on the (R, z≥0) grid using a tensor Simpson rule over (xi,eta)∈[0,1]^2,
    #     with AGAMA-style log mapping and Jacobian.

    #     Integral (half-space build, even in z):
    #         Φ_m(R0,z0) = -G * 2 * ∬_{[0,1]^2}  [ ρ_m(R',z') * Ξ_m(m; R0,z0 | R',z') * (2π R') ] * J(xi,eta)  dxi deta
    #     where:
    #         (R', z', J) = _xieta_to_Rz_jacobian(xi, eta)
    #         J = (∂R/∂xi) * (∂z/∂eta)  (the 2π R' factor is put into the integrand explicitly)

    #     Parameters
    #     ----------
    #     N_int : total target number of samples (rough budget). Used if n_xi/n_eta not given.
    #     n_xi, n_eta : explicit odd counts for Simpson along xi and eta. If given, they override N_int.
    #                 Must be odd and ≥ 3.
    #     m_list : which m to compute (default: 0..mmax).
    #     progress : print per-R line progress.

    #     Returns
    #     -------
    #     dict with the chosen (n_xi, n_eta) and total nodes.
    #     """
    #     if not self._rho_m_interp:
    #         raise RuntimeError("Run compute_rho_m() first.")

    #     # Choose Simpson node counts (odd >=3)
    #     if (n_xi is None) or (n_eta is None):
    #         base = max(9, int(np.sqrt(max(16, N_int))))
    #         if base % 2 == 0:
    #             base += 1
    #         n_xi = base if n_xi is None else int(n_xi)
    #         n_eta = base if n_eta is None else int(n_eta)
    #     if n_xi < 3 or n_xi % 2 == 0:
    #         raise ValueError("Simpson along xi needs odd n_xi >= 3.")
    #     if n_eta < 3 or n_eta % 2 == 0:
    #         raise ValueError("Simpson along eta needs odd n_eta >= 3.")

    #     # Simpson weights on [0,1]
    #     def simpson_weights(n: int) -> np.ndarray:
    #         w = np.ones(n)
    #         w[1:-1:2] = 4.0
    #         w[2:-1:2] = 2.0
    #         w *= (1.0 / (n - 1)) / 3.0   # h = 1/(n-1), scale by h/3
    #         return w

    #     wxi  = simpson_weights(n_xi)
    #     weta = simpson_weights(n_eta)

    #     # Tensor nodes
    #     xi  = np.linspace(0.0, 1.0, n_xi)
    #     eta = np.linspace(0.0, 1.0, n_eta)
    #     XI, ETA = np.meshgrid(xi, eta, indexing="ij")      # (n_xi, n_eta)

    #     # Map to physical (R', z') and get Jacobian part (no 2πR' here)
    #     Rp, zp, Jmap = self._xieta_to_Rz_jacobian(XI, ETA) # (n_xi, n_eta) each

    #     # Precompute the combined 2D Simpson weights
    #     W2D = (wxi[:, None]) * (weta[None, :])             # (n_xi, n_eta)

    #     # Set which m to compute
    #     m_list = list(range(self.mmax + 1)) if m_list is None else list(m_list)

    #     self._Phi_m_grid = {}
    #     self._Phi_m_interp = {}

    #     for m in tqdm(m_list, total = len(m_list)):
    #         Phi = np.zeros((self.NR, self.Z_nonneg.size), dtype=complex)

    #         # Precompute density ρ_m at all (R',z') nodes ONCE per m (vectorized)
    #         rho_grid = self.rho_m_eval(m, Rp, zp)          # (n_xi, n_eta), complex

    #         for i, R0 in enumerate(self.R):
    #             for j, z0 in enumerate(self.Z_nonneg):
    #                 # Kernel at all nodes for this (R0, z0)
    #                 Xi_kernel = self.kernel_Xi_m(m, R0, z0, Rp, zp)   # (n_xi, n_eta), real

    #                 # Build integrand: ρ_m * Ξ_m * (2π R') * Jmap
    #                 F = rho_grid * Xi_kernel * (2.0 * np.pi) * Rp * Jmap  # complex

    #                 # Simpson tensor product on [0,1]^2
    #                 I = np.sum(W2D * F)

    #                 # Even symmetry in z' (upper half only)
    #                 Phi[i, j] = -self.G * 2.0 * I

    #             if progress:
    #                 print(f"[mapped simpson] m={m}  R[{i+1}/{self.NR}]")

    #         # store grid + interpolator
    #         self._Phi_m_grid[m] = Phi
    #         self._Phi_m_interp[m] = RegularGridInterpolator(
    #             (self.R, self.Z_nonneg), Phi, method="cubic",
    #             bounds_error=False, fill_value=None
    #         )

    #     return {
    #         "n_xi": n_xi,
    #         "n_eta": n_eta,
    #         "total_nodes": n_xi * n_eta,
    #         "rule": "simpson([0,1]^2) with log-mapping",
    #     }

    def compute_phi_m_grid_fixed_mapped(
        self,
        *,
        N_int: int = 10_000,
        n_xi: Optional[int] = None,
        n_eta: Optional[int] = None,
        m_list: Optional[List[int]] = None,
        progress: bool = False,
    ):
        """
        Compute Φ_m(R,z) on the (R, z≥0) grid using a tensor Simpson rule over (xi,eta)∈[0,1]^2,
        with AGAMA-style log mapping and Jacobian.

        IMPORTANT: we integrate z'≥0 only, but *sum* kernel contributions from +z' and −z'
        (Xi_plus + Xi_minus) inside the integrand. This replaces the old “×2 at the end”
        and is correct for z≠0 as well as z=0.

            Φ_m(R0,z0) = -G * ∬_{[0,1]^2}  ρ_m(R',z') * [Ξ_m(R0,z0|R',+z') + Ξ_m(R0,z0|R',-z')]
                                    * (2π R') * J(xi,eta)  dxi deta
        """
        if not self._rho_m_real_interp:
            raise RuntimeError("Run compute_rho_m() first.")

        # --- choose Simpson node counts (odd ≥3) from N_int if not provided ---
        if (n_xi is None) or (n_eta is None):
            base = max(9, int(np.sqrt(max(16, N_int))))
            if base % 2 == 0:
                base += 1
            n_xi = base if n_xi is None else int(n_xi)
            n_eta = base if n_eta is None else int(n_eta)
        if n_xi < 3 or n_xi % 2 == 0:
            raise ValueError("Simpson along xi needs odd n_xi ≥ 3.")
        if n_eta < 3 or n_eta % 2 == 0:
            raise ValueError("Simpson along eta needs odd n_eta ≥ 3.")

        # --- Simpson weights on [0,1] ---
        def simpson_weights(n: int) -> np.ndarray:
            w = np.ones(n)
            w[1:-1:2] = 4.0
            w[2:-1:2] = 2.0
            w *= (1.0 / (n - 1)) / 3.0   # h = 1/(n-1), scale by h/3
            return w

        wxi  = simpson_weights(n_xi)
        weta = simpson_weights(n_eta)

        # --- tensor nodes in (xi,eta) and mapped (R', z') with Jacobian ---
        xi  = np.linspace(0.0, 1.0, n_xi)
        eta = np.linspace(0.0, 1.0, n_eta)
        XI, ETA = np.meshgrid(xi, eta, indexing="ij")                    # (n_xi, n_eta)

        Rp, zp, Jmap = self._xieta_to_Rz_jacobian(XI, ETA)               # (n_xi, n_eta) each
        W2D = (wxi[:, None]) * (weta[None, :])                           # Simpson product weights

        # which m to compute
        m_list = list(range(self.mmax + 1)) if m_list is None else list(m_list)

        self._Phi_m_grid = {}
        self._Phi_m_interp = {}

        for m in tqdm(m_list, total = len(m_list)):
            Phi = np.zeros((self.NR, self.Z_nonneg.size), dtype=complex)

            # Precompute density ρ_m at all (R',z') nodes ONCE per m (even in z)
            rho_grid = self.rho_m_eval(m, Rp, zp)                         # (n_xi, n_eta), complex

            for i, R0 in (enumerate(self.R)):
                for j, z0 in (enumerate(self.Z_nonneg)):
                    # kernel from +z' and −z' (sum, not average)
                    Xi_plus  = self.kernel_Xi_m(m, R0, z0, Rp,  zp)       # real
                    Xi_minus = self.kernel_Xi_m(m, R0, z0, Rp, -zp)       # real
                    Xi_sum   = Xi_plus + Xi_minus

                    # integrand: ρ_m * (Ξ+ + Ξ−) * (2π R') * J
                    F = rho_grid * Xi_sum * (2.0 * np.pi) * Rp * Jmap     # complex

                    # Simpson tensor product on [0,1]^2
                    I = np.sum(W2D * F)

                    # no extra ×2: Xi_sum already accounts for both halves in z′
                    Phi[i, j] = -self.G * I

                if progress:
                    print(f"[mapped simpson] m={m}  R[{i+1}/{self.NR}]")

            # store grid + interpolator
            self._Phi_m_grid[m] = Phi
            # self._Phi_m_interp[m] = RegularGridInterpolator(
            #     (self.R, self.Z_nonneg), Phi, method="linear",
            #     bounds_error=False, fill_value=None
            # )
            self._Phi_m_real_interp[m] = CubicSpline2D(
                (self.R, self.Z_nonneg), Phi.real,
                bounds_error=False, fill_value=0.
            )
            self._Phi_m_imag_interp[m] = CubicSpline2D(
                (self.R, self.Z_nonneg), Phi.imag,
                bounds_error=False, fill_value=0.
            )

        return {
            "n_xi": n_xi,
            "n_eta": n_eta,
            "total_nodes": n_xi * n_eta,
            "rule": "simpson([0,1]^2) with log-mapping; Xi(+z')+Xi(−z')",
        }


    # --------- evaluate Φ from Φ_m interpolators ---------
    def phi_m_eval(self, m: int, R: ArrayLike, z: ArrayLike) -> Array:
        if m not in self._Phi_m_real_interp:
            raise RuntimeError("compute_phi_m_grid_*() and build of Φ_m interpolators required first.")
        Rb = np.asarray(R, float); zb = np.abs(np.asarray(z, float))
        # pts = np.stack(np.broadcast_arrays(Rb, zb), axis=-1).reshape(-1, 2)
        shape = Rb.shape
        pts = np.column_stack((Rb.ravel(), zb.ravel()))
        return self._Phi_m_real_interp[m](pts).reshape(shape) + 1j * self._Phi_m_imag_interp[m](pts).reshape(shape)

    def potential(self, R: ArrayLike, z: ArrayLike, phi: ArrayLike) -> Array:
        Rb = np.asarray(R, float); zb = np.asarray(z, float); ph = np.asarray(phi, float)
        Rb, zb, ph = np.broadcast_arrays(Rb, zb, ph)
        # m=0 term
        out = self.phi_m_eval(0, Rb, zb).real
        # m>=1 terms: Φ += 2 * (Re Φ_m * cos mφ  - Im Φ_m * sin mφ)
        for m in range(1, self.mmax+1):
            Phi_m = self.phi_m_eval(m, Rb, zb)
            out += 2.0 * (Phi_m.real * np.cos(m*ph) - Phi_m.imag * np.sin(m*ph))
        return out

    def potential_cartesian(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Array:
        R, phi, zz = cartesian_to_cylindrical(x, y, z)
        return self.potential(R, zz, phi)

    def density(self, R: ArrayLike, z: ArrayLike, phi: ArrayLike) -> Array:
        """
        Reconstruct ρ(R,z,φ) from the complex Fourier moments ρ_m(R,z) that were
        built by compute_rho_m(). Uses even symmetry in z via |z|.
        """
        if not self._rho_m_real_interp:
            raise RuntimeError("compute_rho_m() must be called before density().")

        Rb = np.asarray(R, float)
        zb = np.fabs(np.asarray(z, float))
        ph = np.asarray(phi, float)

        # m = 0
        rho = self.rho_m_eval(0, Rb, zb).real

        # m >= 1
        for m in range(1, self.mmax + 1):
            rhom = self.rho_m_eval(m, Rb, zb)
            rho += (2.0 * rhom * np.exp(1j*m*ph)).real#(rhom.real * np.cos(m * ph) - rhom.imag * np.sin(m * ph))
        return rho

    def density_cartesian(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Array:
        """Reconstruct ρ(x,y,z) from stored ρ_m by converting to cylindrical first."""
        R, phi, zz = cartesian_to_cylindrical(x, y, z)
        return self.density(R, zz, phi)



def cartesian_to_cylindrical(x: Array, y: Array, z: Array) -> Tuple[Array, Array, Array]:
    """
    Convert Cartesian (x, y, z) -> Cylindrical (R, φ, z).

    Conventions
    -----------
    R >= 0
    φ ∈ [0, 2π) measured from +x toward +y (radians)
    z unchanged

    Notes
    -----
    - Fully vectorized: x,y,z can be scalars or same-shaped arrays.
    - At R=0, φ is set to 0 by convention.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    R  = np.sqrt(x*x + y*y)
    phi  = np.arctan2(y, x)            # (-π, π]
    phi  = np.where(phi < 0.0, phi + 2*np.pi, phi)  # [0, 2π)

    # define φ=0 on the axis to avoid NaNs (purely conventional)
    phi  = np.where(R == 0.0, 0.0, phi)
    return R, phi, z

def cylindrical_to_cartesian(R: Array, phi: Array, z: Array) -> Tuple[Array, Array, Array]:
    """
    Convert Cylindrical (R, φ, z) -> Cartesian (x, y, z).

    Conventions
    -----------
    R >= 0
    φ in radians (any real value is accepted; it is periodic)
    z unchanged

    Notes
    -----
    - Fully vectorized: R,φ,z can be scalars or same-shaped arrays.
    """
    R = np.asarray(R, dtype=float)
    phi = np.asarray(phi, dtype=float)
    z = np.asarray(z, dtype=float)

    x = R * np.cos(phi)
    y = R * np.sin(phi)
    return x, y, z

