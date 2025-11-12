# """
# AGAMA-style potential expansions in pure Python: Multipole and CylSpline.

# Implements two classes that mirror the philosophy of AGAMA (Vasiliev) expansions:
# - MultipoleExpansion: spherical-harmonic expansion with separate lmax/mmax, automatic
#   log-radius grid selection, scaling of amplitudes, and quintic(-ish) spline interpolation
#   in ln r. Solves Poisson's equation from an arbitrary density ρ(r,θ,φ) via the exact
#   1D Green-function integrals for each (l,m), cf. Eq. (31) and the angular moments Eq. (32).
# - CylSplineExpansion: azimuthal Fourier expansion in φ with 2D spline in transformed
#   (R̃, z̃) = (ln(1+R/R0), ln(1+z/R0)) coordinates. Constructed from a density profile via
#   the cylindrical Green's function (Bessel integral kernel). This is computationally
#   expensive but done once; afterwards, evaluation is fast.

# This module follows the AGAMA documentation for Multipole and CylSpline, including:
# - Separate lmax and mmax (m0(l) = min(l, mmax)).
# - Logarithmic radial grid centered at a fiducial radius r* determined from the spherically
#   averaged density (curvature criterion), with near-constant ratio f between nodes and
#   automatic inner/outer clipping.
# - Amplitude scalings: for Multipole, the main l=m=0 term is stored with a log-scaling of
#   its amplitude, and other (l,m) are normalized to the l=0 term. (Scaling is disabled if
#   signs change.) For CylSpline, analogous scaling in 2D.
# - Extrapolation with power-law asymptotes at the radial grid boundaries for Multipole.

# References: see AGAMA manual, Appendix A.4.1 (Multipole) and A.4.2 (CylSpline).
# """
# from __future__ import annotations
# import numpy as np
# from numpy.polynomial.legendre import leggauss
# from dataclasses import dataclass
# from typing import Callable, Optional, Tuple, Dict
# from scipy.special import sph_harm, jv
# from scipy.integrate import simpson, quad
# from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

# Array = np.ndarray

# # ----------------------------- Utilities -----------------------------

# def _legendre_gl(n: int) -> Tuple[Array, Array]:
#     x, w = leggauss(n)
#     theta = np.arccos(x)
#     return theta, w

# @dataclass
# class AngularQuad:
#     n_theta: int = 40
#     n_phi: int = 80

#     def nodes(self) -> Tuple[Array, Array, Array]:
#         theta, w_th = _legendre_gl(self.n_theta)
#         phi = np.linspace(0.0, 2*np.pi, self.n_phi, endpoint=False)
#         w_phi = (2*np.pi) / self.n_phi
#         W = (w_th[:, None] * w_phi)  # (n_theta, 1) -> broadcast
#         W = np.repeat(W, self.n_phi, axis=1)
#         TH, PH = np.meshgrid(theta, phi, indexing='ij')
#         return TH, PH, W

# # ------------------------- Radial grid chooser ------------------------

# def _spherical_average_density(r: Array, rho_func: Callable[[Array, Array, Array], Array], ang: AngularQuad) -> Array:
#     TH, PH, W = ang.nodes()
#     out = np.empty_like(r)
#     for i, ri in enumerate(r):
#         rr = np.full_like(TH, ri, dtype=float)
#         out[i] = (np.sum(rho_func(rr, TH, PH) * W) / (4*np.pi))
#     return out

# def _choose_log_grid_from_density(r_guess: Array, rho00: Array, NR: int = 25) -> Array:
#     """Choose a log-radius grid centered at r* using a curvature criterion on ln rho00.
#     r_guess should span the region of interest (e.g. 1e-6..1e6 in units of scale)."""
#     r = np.asarray(r_guess)
#     # Work in log-space
#     lr = np.log(r)
#     # Avoid zeros
#     rho_pos = np.clip(rho00, 1e-300, None)
#     ln_rho = np.log(rho_pos)
#     # Second derivative wrt ln r
#     d1 = np.gradient(ln_rho, lr)
#     d2 = np.gradient(d1, lr)
#     weight = r**2 * rho_pos
#     score = np.abs(d2) * weight
#     k = int(np.argmax(score))
#     r_star = r[k]
#     # Set a geometric spacing factor f so that r spans ~D with NR nodes
#     # Aim for D~1e6 for NR~25 as in AGAMA docs
#     D_target = 1e6
#     f = D_target**(1.0/(NR-1))
#     # Build symmetric grid around r* and then clip to r range
#     half = (NR-1)//2
#     idx = np.arange(-half, half+1)
#     grid = r_star * (f**idx)
#     # If NR is even, extend one more outward point to keep length NR
#     if grid.size < NR:
#         grid = np.append(grid, grid[-1]*f)
#     # Ensure strictly increasing and positive
#     grid = np.unique(np.maximum(grid, np.min(r[r>0])))
#     return grid

# # ---------------------------- Multipole -------------------------------

# class MultipoleExpansion:
#     def __init__(
#         self,
#         rho: Callable[[Array, Array, Array], Array],
#         lmax: int,
#         mmax: int,
#         G: float = 1.0,
#         NR: int = 25,
#         rmin: Optional[float] = None,
#         rmax: Optional[float] = None,
#         ang: Optional[AngularQuad] = None,
#     ):
#         self.rho = rho
#         self.lmax = int(lmax)
#         self.mmax = int(mmax)
#         self.G = float(G)
#         self.NR = int(NR)
#         self.ang = ang or AngularQuad()
#         # provisional wide grid for spherical average to choose final grid
#         r_probe = np.geomspace(1e-6, 1e6, 400)
#         rho00_probe = _spherical_average_density(r_probe, rho, self.ang)
#         grid_auto = _choose_log_grid_from_density(r_probe, rho00_probe, NR=self.NR)
#         if rmin is not None:
#             grid_auto[0] = max(grid_auto[0], float(rmin))
#         if rmax is not None:
#             grid_auto[-1] = min(grid_auto[-1], float(rmax))
#         self.r = grid_auto
#         # compute angular moments ρ_{l m}(r)
#         self._rho_lm: Dict[Tuple[int,int], Array] = {}
#         self._phi_lm_spline: Dict[Tuple[int,int], InterpolatedUnivariateSpline] = {}
#         self._build_rho_lm()
#         self._solve_poisson_and_fit()

#     def _build_rho_lm(self):
#         TH, PH, W = self.ang.nodes()
#         for l in range(self.lmax+1):
#             mcap = min(l, self.mmax)
#             Pnorm = np.sqrt((2*l+1)/(4*np.pi))
#             P_lm_cache: Dict[int, Array] = {}
#             # evaluate associated Legendre via real Y_lm combos
#             for m in range(-mcap, mcap+1):
#                 acc = np.empty_like(self.r)
#                 # real basis: trig_mφ * P~_l^m(cosθ) with normalization as in AGAMA
#                 # We'll use complex Y_lm and then take real/imag to map to cos/sin parts.
#                 Ylm_conj = np.conjugate(sph_harm(m, l, PH, TH))
#                 trig = None  # encoded in real/imag parts
#                 for i, ri in enumerate(self.r):
#                     rr = np.full_like(TH, ri, dtype=float)
#                     rho_vals = self.rho(rr, TH, PH)
#                     # complex harmonic moment (standard): a_lm = ∫ Y*_lm rho dΩ
#                     alm = np.sum(Ylm_conj * rho_vals * W)
#                     # Map to AGAMA's real coefficients φ_{l,m} basis later by combining ±m
#                     acc[i] = alm.real  # store real part temporarily; we'll reconstruct below
#                 # Store complex coefficient path: for robust reconstruction keep both parts
#                 self._rho_lm[(l,m)] = acc  # Note: represents Re[ρ_lm^complex]; used for φ via integrals
#             # Note: For strict AGAMA real-harmonic separation, one would transform here.

#     def _solve_poisson_and_fit(self):
#         r = self.r
#         ln_r = np.log(r)
#         for l in range(self.lmax+1):
#             mcap = min(l, self.mmax)
#             for m in range(-mcap, mcap+1):
#                 rho_lm = self._rho_lm[(l,m)]
#                 # Green-function integrals (Eq. 31):
#                 f_inner = rho_lm * r**(l+2)
#                 I_inner = np.empty_like(r)
#                 I_inner[0] = 0.0
#                 for k in range(1, len(r)):
#                     I_inner[k] = simpson(f_inner[:k+1], r[:k+1])
#                 f_outer = rho_lm * r**(1-l)
#                 I_outer = np.empty_like(r)
#                 I_outer[-1] = 0.0
#                 for k in range(len(r)-2, -1, -1):
#                     I_outer[k] = simpson(f_outer[k:], r[k:])
#                 pref = -4*np.pi*self.G/(2*l+1)
#                 phi_lm = pref * (r**(-(l+1))*I_inner + r**l * I_outer)
#                 # Fit spline in ln r (k=5 ~ quintic)
#                 # Use small smoothing to stabilize if needed
#                 spline = InterpolatedUnivariateSpline(ln_r, phi_lm, k=min(5, len(r)-1))
#                 self._phi_lm_spline[(l,m)] = spline
#         # record Φ00 sign behavior for optional amplitude scaling (not strictly needed for evaluation)
#         self._phi00_ln = self._phi_lm_spline[(0,0)](ln_r)

#     def potential(self, r: Array, theta: Array, phi: Array) -> Array:
#         r = np.asarray(r)
#         th = np.asarray(theta)
#         ph = np.asarray(phi)
#         R, TH, PH = np.broadcast_arrays(r, th, ph)
#         lnR = np.log(np.clip(R, self.r[0], self.r[-1]))
#         out = np.zeros_like(R, dtype=float)
#         for l in range(self.lmax+1):
#             mcap = min(l, self.mmax)
#             for m in range(-mcap, mcap+1):
#                 Phi_lm = self._phi_lm_spline[(l,m)](lnR)
#                 Ylm = sph_harm(m, l, PH, TH)
#                 out += (Phi_lm * Ylm).real
#         return out

# # ---------------------------- CylSpline -------------------------------

# class CylSplineExpansion:
#     def __init__(
#         self,
#         rho: Callable[[Array, Array, Array], Array],  # ρ(R,z,φ)
#         mmax: int,
#         R0: Optional[float] = None,
#         NR: int = 25,
#         Nz: int = 25,
#         Rmin: Optional[float] = None,
#         Rmax: Optional[float] = None,
#         Zmax: Optional[float] = None,
#         G: float = 1.0,
#     ):
#         self.rho = rho
#         self.mmax = int(mmax)
#         self.NR = int(NR)
#         self.Nz = int(Nz)
#         self.G = float(G)
#         # choose grids (rough heuristic; user can override by passing limits)
#         Rmin = 0.0 if Rmin is None else float(Rmin)
#         Rmax = 50.0 if Rmax is None else float(Rmax)
#         Zmax = 50.0 if Zmax is None else float(Zmax)
#         self.Rgrid = np.geomspace(max(Rmin, 1e-3), Rmax, NR)
#         self.Zgrid = np.geomspace(1e-3, Zmax, Nz)
#         # symmetric z grid
#         self.Zgrid = np.unique(np.sort(np.concatenate((-self.Zgrid[::-1], [0.0], self.Zgrid))))
#         # transform coordinates
#         self.R0 = (self.Rgrid[self.NR//2] if R0 is None else float(R0))
#         self._Rtilde = np.log1p(self.Rgrid/self.R0)
#         self._Ztilde = np.log1p(np.abs(self.Zgrid)/self.R0) * np.sign(self.Zgrid)
#         # compute φ-harmonics of potential on grid via cylindrical Green's function integral (Eq. 33)
#         self._phi_m_spline: Dict[int, RectBivariateSpline] = {}
#         self._build_phim_splines()

#     def _kernel_m(self, m: int, R: float, z: float, Rp: Array, zp: Array) -> Array:
#         # Ξ_m = ∫_0^∞ dk J_m(kR) J_m(kRp) exp(-k|z-zp|)
#         # Numerically integrate over k for each (Rp,zp). This is costly; use adaptive quad.
#         Rp = np.asarray(Rp)
#         zp = np.asarray(zp)
#         arr = np.empty_like(Rp, dtype=float)
#         for i in range(Rp.size):
#             Rpi = float(Rp.flat[i])
#             zpi = float(zp.flat[i])
#             def integrand(k):
#                 return jv(m, k*R) * jv(m, k*Rpi) * np.exp(-k*abs(z - zpi))
#             # heuristic upper limit based on decay scale
#             scale = 1.0 / max(1e-6, R + Rpi + abs(z-zpi))
#             kmax = 50.0/scale
#             val, _ = quad(integrand, 0.0, kmax, epsabs=1e-6, epsrel=1e-5, limit=200)
#             arr.flat[i] = val
#         return arr

#     def _build_phim_splines(self):
#         Rg, Zg = np.meshgrid(self.Rgrid, self.Zgrid, indexing='ij')
#         # precompute density samples on a helper grid in φ
#         nphi = 64
#         phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
#         dphi = 2*np.pi/nphi
#         # Fourier project ρ_m(R',z') = ∫ dφ' ρ e^{-i m φ'}
#         rho_m = {m: np.zeros_like(Rg) for m in range(0, self.mmax+1)}
#         for iphi, ph in enumerate(phi):
#             rho_vals = self.rho(Rg, Zg, ph)
#             for m in range(0, self.mmax+1):
#                 rho_m[m] += rho_vals * np.cos(m*ph) * dphi  # use cosine only (real even part)
#         # Compute φ_m(R,z) on the grid via double integral over R',z' with kernel Ξ_m
#         for m in range(0, self.mmax+1):
#             Phi = np.zeros_like(Rg, dtype=float)
#             for iR, R in enumerate(self.Rgrid):
#                 for iz, z in enumerate(self.Zgrid):
#                     # integrate over R',z' using trapezoid/Simpson separable measure 2π R' ρ_m Ξ_m
#                     # Build flat arrays for integration nodes
#                     Rp = self.Rgrid
#                     Zp = self.Zgrid
#                     # separable measure: sum over grid with weights
#                     wR = np.ones_like(Rp); wR[1:-1] = 2.0; wR *= (Rp[1]-Rp[0])  # crude trap weights in R (non-uniform! will adjust)
#                     # Better: integrate in ln R with simpson; approximate by simpson on actual R
#                     Phi_sum = 0.0
#                     for jR, Rp_j in enumerate(Rp):
#                         # z integral
#                         kern = self._kernel_m(m, R, z, np.full_like(Zp, Rp_j), Zp)
#                         integrand_z = 2*np.pi * Rp_j * rho_m[m][jR,:] * kern
#                         Iz = simpson(integrand_z, Zp)
#                         Phi_sum += Iz
#                     Phi[iR, iz] = -self.G * Phi_sum
#             # build 2D spline in (R̃, Z̃)
#             Rtil = np.log1p(self.Rgrid/self.R0)
#             Ztil = np.log1p(np.abs(self.Zgrid)/self.R0) * np.sign(self.Zgrid)
#             self._phi_m_spline[m] = RectBivariateSpline(Rtil, Ztil, Phi, kx=3, ky=3, s=0.0)

#     def potential(self, R: Array, z: Array, phi: Array) -> Array:
#         R = np.asarray(R); z = np.asarray(z); ph = np.asarray(phi)
#         Rb, zb, phb = np.broadcast_arrays(R, z, ph)
#         Rtil = np.log1p(Rb/self.R0)
#         Ztil = np.log1p(np.abs(zb)/self.R0) * np.sign(zb)
#         out = np.zeros_like(Rb, dtype=float)
#         for m in range(0, self.mmax+1):
#             Phi_m = self._phi_m_spline[m].ev(Rtil, Ztil)
#             out += Phi_m * np.cos(m*phb)
#         return out

"""
AGAMA-style potential expansions in pure Python (readable, heavily commented).

This module implements two *data-driven* gravitational potential expansions
inspired by AGAMA (Vasiliev):

1) MultipoleExpansion (spherical-harmonic):
   - Expand the *angular* dependence of the potential in spherical harmonics Y_{l m}.
   - For each (l,m), solve the *radial* part exactly from a supplied mass density ρ
     using the Green-function identities for Poisson's equation in spherical coords.
   - Store Φ_{l m}(r) on a log-radius grid and interpolate (spline) in ln r.
   - You control lmax and mmax independently; we do NOT cull by symmetry.

2) CylSplineExpansion (cylindrical Fourier + 2D spline):
   - Expand the azimuthal dependence in a cosine Fourier series (m = 0..mmax).
   - For each m, construct Φ_m(R,z) on a 2D grid in transformed coordinates
     (R~, z~) = (ln(1+R/R0), ln(1+|z|/R0)*sign z), then spline-interpolate.
   - The construction uses the cylindrical Green-function kernel written as a
     Bessel-k integral of J_m(kR) J_m(kR') exp(-k|z-z'|). This is computationally
     heavy but executed once; evaluation afterwards is fast.

The code here is deliberately *verbose and didactic* to make porting to JAX, C++,
Julia, etc., straightforward. The focus is clarity of algorithmic steps rather than
micro-optimizations (those can be added later).

References (match your PDF/manual sections by idea, not verbatim labels):
- Multipole expansion: Poisson Green function in spherical harmonics, e.g.
  Binney & Tremaine (2008) §2.5–2.7.
- Cylindrical Fourier (CylSpline-like): Hankel/Bessel representations of the
  cylindrical Green function; see standard potential theory texts and AGAMA's
  documentation (Legendre-Q and asymptotic refinements are possible later).

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict
from tqdm import tqdm

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import simpson, quad
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.special import sph_harm, jv

Array = np.ndarray

# -----------------------------------------------------------------------------
#                           QUADRATURE / ANGULAR GRID
# -----------------------------------------------------------------------------

@dataclass
class AngularQuad:
    """Angular quadrature settings for θ and φ.

    We integrate over solid angle dΩ = sinθ dθ dφ. Using x = cosθ, dΩ = dφ d(cosθ).
    - For θ: Gaussian quadrature in x=cosθ (Gauss–Legendre) integrates polynomials
      in x exactly and is standard for spherical harmonics.
    - For φ: uniform trapezoidal rule is spectrally accurate for periodic functions.

    Parameters
    ----------
    n_theta : number of Gauss–Legendre nodes in θ (mapped from x=cosθ)
    n_phi   : number of equispaced nodes in φ ∈ [0, 2π)
    """
    n_theta: int = 40
    n_phi: int = 80

    def nodes(self) -> Tuple[Array, Array, Array]:
        """Return angular nodes and the 2D quadrature weights.

        Returns
        -------
        TH : (n_theta, n_phi) 2D array of θ nodes (radians)
        PH : (n_theta, n_phi) 2D array of φ nodes (radians)
        W  : (n_theta, n_phi) weights so that sum(F(TH,PH)*W) ≈ ∫ F dΩ
        """
        # -- Gauss–Legendre in x = cosθ: ∫_{-1}^1 f(x) dx ≈ Σ w_i f(x_i)
        x_nodes, w_x = leggauss(self.n_theta)
        theta = np.arccos(x_nodes)              # map x -> θ
        phi = np.linspace(0.0, 2*np.pi, self.n_phi, endpoint=False)
        # -- dΩ = dφ d(cosθ). GL already incorporates the Jacobian for θ-part; φ uses trapezoid.
        w_phi = (2*np.pi) / self.n_phi          # each φ-panel has equal weight
        # Tile to a 2D weight field W_{ij} = w_x[i] * w_phi
        W = (w_x[:, None] * w_phi)
        W = np.repeat(W, self.n_phi, axis=1)    # shape (n_theta, n_phi)
        # Broadcast θ,φ to 2D grids for vectorized evaluation
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        return TH, PH, W

# -----------------------------------------------------------------------------
#                  SPHERICAL-AVERAGE & RADIAL GRID CONSTRUCTION
# -----------------------------------------------------------------------------

def _spherical_average_density(
    r: Array,
    rho_sph: Callable[[Array, Array, Array], Array],
    ang: AngularQuad,
) -> Array:
    """Compute the spherical average ρ₀₀(r) = (1/4π) ∫ ρ(r,θ,φ) dΩ on a set of radii.

    This is used to (a) gauge the dynamic range and (b) choose a sensible *log-radius*
    grid with approximately uniform information content (akin to AGAMA's recipe).
    """
    TH, PH, W = ang.nodes()
    rho00 = np.empty_like(r, dtype=float)
    for i, ri in enumerate(r):
        rr = np.full_like(TH, float(ri))  # broadcast ri over the angular grid
        rho00[i] = np.sum(rho_sph(rr, TH, PH) * W) / (4*np.pi)
    return rho00


def _choose_log_grid_from_density(
    r_probe: Array,
    rho00: Array,
    NR: int = 25,
) -> Array:
    """Choose a log-radius grid {r_j} based on the curvature of ln ρ₀₀.

    High-level idea (AGAMA-inspired):
    1) Work in ln r. The *curvature* of ln ρ₀₀(ln r) highlights where the structure
       changes the most (core/scale radius/outer rollover). Find r* near the peak of
       |d² ln ρ₀₀ / d(ln r)²|, optionally weighted by r²ρ (emphasize mass-rich zones).
    2) Build a geometric progression around r* with a constant factor f so that NR
       nodes cover a target dynamic range. NR is small (≲25) for compact splines.

    This function is intentionally simple and transparent for portability.
    """
    r = np.asarray(r_probe, dtype=float)
    lr = np.log(r)
    # Prevent log(0) and maintain numerical stability
    rho_pos = np.clip(rho00, 1e-300, None)
    ln_rho = np.log(rho_pos)

    # First and second derivatives with respect to ln r (finite-difference)
    d1 = np.gradient(ln_rho, lr)
    d2 = np.gradient(d1, lr)

    # Optional weight: emphasize where mass lives; crude but effective.
    weight = (r**2) * rho_pos
    score = np.abs(d2) * weight
    k = int(np.argmax(score))
    r_star = float(r[k])

    # Constant ratio f for a desired dynamic range D ≈ f^{NR-1}.
    # The concrete number here is a heuristic; adjust NR to taste.
    D_target = 1e6
    f = D_target ** (1.0 / max(1, NR - 1))

    # Center the grid at r* in log-space: r_j = r* * f^{j - mid}
    half = (NR - 1) // 2
    idx = np.arange(-half, half + 1)
    grid = r_star * (f ** idx)
    # If NR is even, add one extra outer point to keep the requested length
    if grid.size < NR:
        grid = np.append(grid, grid[-1] * f)

    # Ensure strictly increasing and positive values
    grid = np.asarray(grid, dtype=float)
    grid = np.maximum(grid, np.min(r[r > 0]))
    # Unique/monotonic (defensive)
    grid = np.unique(grid)
    return grid

# -----------------------------------------------------------------------------
#                             MULTIPOLE EXPANSION
# -----------------------------------------------------------------------------

class MultipoleExpansion:
    """Spherical-harmonic (l,m) expansion of the gravitational potential.

    Construction pipeline (build-time, executed once):
      Step M1: Choose a *logarithmic r-grid* from a coarse probe of the spherical
               average ρ₀₀ (see functions above). Users can override bounds/NR.
      Step M2: For each (l,m) with 0 ≤ l ≤ lmax and −min(l,mmax) ≤ m ≤ +min(l,mmax),
               compute the *angular harmonic moments* ρ_{l m}(r) = ∫ Y*_{l m} ρ dΩ
               at each grid radius via quadrature over (θ,φ).
      Step M3: For each (l,m), solve the *radial Green integrals* exactly to obtain
               Φ_{l m}(r) from ρ_{l m}(r):
                 Φ_{l m}(r) = -(4πG)/(2l+1) [ r^{-(l+1)} ∫_0^r ρ_{l m}(r') r'^{l+2} dr'
                                            + r^{l} ∫_r^∞ ρ_{l m}(r') r'^{1-l} dr' ].
               (We truncate ∞ at the last node; adding an outer taper/asymptote is
                straightforward if needed.)
      Step M4: Build an *interpolant* Φ_{l m}(ln r) (univariate spline) so that later
               evaluations are O(#harmonics) and very fast.

    Evaluation (runtime):
      Step M5: Given (r,θ,φ), broadcast inputs, evaluate all Φ_{l m}(ln r) and sum
               Φ(r,θ,φ) = Σ_{l,m} Φ_{l m}(r) Y_{l m}(θ,φ), taking the real part.

    Notes
    -----
    - We do *not* prune by symmetry; this matches the user's request to keep all
      coefficients up to mmax (and obviously m ≤ l).
    - We use complex Y_{l m} from SciPy and take Re[...]; porting to a real basis
      (cos mφ / sin mφ times associated Legendre) is a mechanical transformation.
    - Interpolation is performed in ln r, which is robust over wide dynamic ranges.
    - Force computation (∇Φ) can be added by differentiating the spline and the
      spherical harmonics; left out here for clarity.
    """

    def __init__(
        self,
        rho: Callable[[Array, Array, Array], Array],
        lmax: int,
        mmax: int,
        G: float = 1.0,
        NR: int = 25,
        rmin: Optional[float] = None,
        rmax: Optional[float] = None,
        ang: Optional[AngularQuad] = None,
    ):
        self.rho = rho
        self.lmax = int(lmax)
        self.mmax = int(mmax)
        self.G = float(G)
        self.NR = int(NR)
        self.ang = ang or AngularQuad()

        # --- Step M1: choose the logarithmic radial grid from a wide probe ---
        r_probe = np.geomspace(1e-6, 1e6, 400)  # wide, unitless-ish range; OK to change
        rho00_probe = _spherical_average_density(r_probe, rho, self.ang)
        # grid_auto = _choose_log_grid_from_density(r_probe, rho00_probe, NR=self.NR)
        # print('grid_auto', grid_auto)
        # # Respect optional user clamps (keep monotonic)
        # if rmin is not None:
        #     grid_auto[0] = max(grid_auto[0], float(rmin))
        # if rmax is not None:
        #     grid_auto[-1] = min(grid_auto[-1], float(rmax))
        rmin = 1e-3 if rmin is None else float(rmin)
        rmax = 1e3 if rmax is None else float(rmax)
        grid_auto = np.geomspace(rmin, rmax, self.NR)
        self.r = np.asarray(grid_auto, dtype=float)

        # Storage for harmonic moments and splines
        self._rho_lm: Dict[Tuple[int, int], Array] = {}
        self._phi_lm_spline_re: Dict[Tuple[int, int], InterpolatedUnivariateSpline] = {}
        self._phi_lm_spline_im: Dict[Tuple[int, int], InterpolatedUnivariateSpline] = {}

        # Build-time steps
        self._build_rho_lm()               # Step M2
        self._solve_poisson_and_fit()      # Steps M3–M4


    def _build_rho_lm(self) -> None:
        """Step M2: compute angular harmonic moments ρ_{l m}(r) on the r-grid.

        ρ_{l m}(r) = ∫ Y*_{l m}(θ,φ) ρ(r,θ,φ) dΩ ≈ SUM_{θ,φ} Y*_{l m}(TH,PH) ρ(ri,TH,PH) W.
        **Keep complex values**; both Re and Im are physical and map to cos/sin sectors.
        """
        TH, PH, W = self.ang.nodes()  # angular nodes + weights
        for l in range(self.lmax + 1):
            mcap = min(l, self.mmax)
            for m in range(-mcap, mcap + 1):
                Ylm_conj = np.conjugate(sph_harm(m, l, PH, TH))
                rho_lm = np.empty_like(self.r, dtype=complex)
                for i, ri in enumerate(self.r):
                    rr = np.full_like(TH, float(ri))
                    xx, yy, zz = sph_to_cart(rr, TH, PH)
                    rho_vals = self.rho(xx, yy, zz)
                    rho_lm[i] = np.sum(Ylm_conj * rho_vals * W)  # complex
                self._rho_lm[(l, m)] = rho_lm

    def _solve_poisson_and_fit(self) -> None:
        """Steps M3–M4: solve radial integrals for Φ_{l m} and fit splines in ln r.

        We compute I_in/I_out using complex arithmetic and then store separate
        splines for Re[Φ_{l m}] and Im[Φ_{l m}] against ln r for numerical stability.
        """
        r = self.r
        ln_r = np.log(r)
        for l in range(self.lmax + 1):
            mcap = min(l, self.mmax)
            for m in range(-mcap, mcap + 1):
                rho_lm = self._rho_lm[(l, m)]  # complex array

                # Inner integral I_in up to each r_k (prefix Simpson)
                integrand_inner = rho_lm * (r ** (l + 2))
                I_in = np.empty_like(r, dtype=complex)
                I_in[0] = 0.0 + 0.0j
                for k in range(1, len(r)):
                    I_in[k] = simpson(integrand_inner[: k + 1], r[: k + 1])

                # Outer integral I_out from each r_k to r_max (suffix Simpson)
                integrand_outer = rho_lm * (r ** (1 - l))
                I_out = np.empty_like(r, dtype=complex)
                I_out[-1] = 0.0 + 0.0j
                for k in range(len(r) - 2, -1, -1):
                    I_out[k] = simpson(integrand_outer[k:], r[k:])

                pref = -4.0 * np.pi * self.G / (2 * l + 1)
                phi_lm = pref * ( (r ** (-(l + 1))) * I_in + (r ** l) * I_out )  # complex

                # Fit separate splines for real and imaginary parts vs ln r
                k_spline = min(5, len(r) - 1)
                
                self._phi_lm_spline_re[(l, m)] = InterpolatedUnivariateSpline(ln_r, phi_lm.real, k=k_spline)
                self._phi_lm_spline_im[(l, m)] = InterpolatedUnivariateSpline(ln_r, phi_lm.imag, k=k_spline)

    def potential(self, r: Array, theta: Array, phi: Array) -> Array:
        """Step M5: evaluate Φ(r,θ,φ) by summing Re[ Φ_{l m}(r) Y_{l m}(θ,φ) ].

        We reconstruct complex Φ_{l m} from its Re/Im splines at ln r and take the
        real part of the harmonic sum. This preserves all m≠0 information.
        """
        r = np.asarray(r, dtype=float)
        th = np.asarray(theta, dtype=float)
        ph = np.asarray(phi, dtype=float)
        R, TH, PH = np.broadcast_arrays(r, th, ph)

        lnR = np.log(np.clip(R, self.r[0], self.r[-1]))
        out = np.zeros_like(R, dtype=float)

        for l in range(self.lmax + 1):
            mcap = min(l, self.mmax)
            for m in range(-mcap, mcap + 1):
                re = self._phi_lm_spline_re[(l, m)](lnR)
                im = self._phi_lm_spline_im[(l, m)](lnR)
                Phi_lm = re + 1j * im
                Ylm = sph_harm(m, l, PH, TH)
                out += (Phi_lm * Ylm).real
        return out

    # Optional diagnostic: power in each (l,m) at a reference radius (first grid node)
    def lm_power_snapshot(self) -> Dict[Tuple[int,int], float]:
        """Return |Φ_{l m}(r₁)| for quick diagnostics of non-axisymmetric content."""
        ln_r1 = np.log(self.r[max(1, len(self.r)//3)])
        power = {}
        for l in range(self.lmax + 1):
            mcap = min(l, self.mmax)
            for m in range(-mcap, mcap + 1):
                re = float(self._phi_lm_spline_re[(l, m)](ln_r1))
                im = float(self._phi_lm_spline_im[(l, m)](ln_r1))
                power[(l, m)] = np.hypot(re, im)
        return power

    def _phi_lm_and_u_derivs(self, lnR):
        """
        Internal: evaluate Phi_lm, dPhi/du, d2Phi/du2 for all (l,m) at u=lnR.
        Returns a dict (l,m) -> (Phi_lm_complex, d1_u_complex, d2_u_complex).
        """
        out = {}
        for l in range(self.lmax + 1):
            mcap = min(l, self.mmax)
            for m in range(-mcap, mcap + 1):
                s_re = self._phi_lm_spline_re[(l, m)]
                s_im = self._phi_lm_spline_im[(l, m)]
                # values
                v0 = s_re(lnR) + 1j * s_im(lnR)
                # first and second derivatives wrt u=ln r
                v1 = s_re.derivative(1)(lnR) + 1j * s_im.derivative(1)(lnR)
                v2 = s_re.derivative(2)(lnR) + 1j * s_im.derivative(2)(lnR)
                out[(l, m)] = (v0, v1, v2)
        return out

    def density(self, r, theta, phi):
        """
        Recover rho(r,theta,phi) from the stored multipole potential via Poisson's equation:
            rho = (1/(4πG)) * sum_{l,m} [ (Phi'_u + Phi''_u)/r^2 - l(l+1)Phi/r^2 ] * Y_lm
        where ' denotes derivative w.r.t. u = ln r.
        """
        r = np.asarray(r, dtype=float)
        th = np.asarray(theta, dtype=float)
        ph = np.asarray(phi, dtype=float)
        R, TH, PH = np.broadcast_arrays(r, th, ph)

        # evaluate u = ln r inside the tabulated bounds
        lnR = np.log(np.clip(R, self.r[0], self.r[-1]))

        # prefetch all harmonics at these radii
        lm_vals = self._phi_lm_and_u_derivs(lnR)

        # assemble the sum
        out = np.zeros_like(R, dtype=float)
        r2 = R**2
        inv_r2 = np.where(r2 > 0, 1.0 / r2, 0.0)

        for l in range(self.lmax + 1):
            mcap = min(l, self.mmax)
            ll = l * (l + 1)
            for m in range(-mcap, mcap + 1):
                Phi_lm, d1u, d2u = lm_vals[(l, m)]
                # radial operator in terms of u-derivatives
                lap_rad = inv_r2 * (d1u + d2u - ll * Phi_lm)
                Ylm = sph_harm(m, l, PH, TH)
                out += (lap_rad * Ylm).real

        return out / (4.0 * np.pi * self.G)

    def density_cartesian(self, x, y, z):
        """
        Convenience: rho(x,y,z) using the spherical-harmonic reconstruction.
        """
        r, th, ph = cart_to_sph(x, y, z)
        return self.density(r, th, ph)
# -----------------------------------------------------------------------------
#                           CYLINDRICAL SPLINE EXPANSION
# -----------------------------------------------------------------------------

def rho_from_potential_cartesian(
    phi: Callable[[Array, Array, Array], Array],
    x: Array,
    y: Array,
    z: Array,
    *,
    G: float = 1.0,
    order: int = 2,          # 2 or 4 (central difference order)
    h: Optional[float] = None,
    hx: Optional[Array] = None,
    hy: Optional[Array] = None,
    hz: Optional[Array] = None,
    rel_step: float = 1e-4,  # used if h/hx/hy/hz are not provided
    abs_step: float = 1e-6   # small floor on step size
) -> Array:
    """
    Generic Φ→ρ via Poisson in Cartesian coords:
        rho = (1/(4πG)) * ( Φ_xx + Φ_yy + Φ_zz )

    Parameters
    ----------
    phi : callable Φ(x,y,z) -> array (broadcasted to x/y/z shape)
    x,y,z : arrays; can be scalars, same-shape arrays (scattered), or meshgrids.
    G : gravitational constant in your unit system.
    order : 2 or 4, for 2nd- or 4th-order central-difference Laplacian.
    h : scalar step used for all axes if hx/hy/hz not given.
    hx,hy,hz : per-axis steps; can be scalars or arrays broadcastable to x/y/z.
    rel_step, abs_step : used to auto-pick steps if none are given.

    Returns
    -------
    rho : array of same shape as broadcast(x,y,z)
    """
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    Z = np.asarray(z, dtype=float)
    # Broadcast to common shape
    X, Y, Z = np.broadcast_arrays(X, Y, Z)

    # Choose step sizes if not provided.
    # If h given -> use for all axes. Otherwise adaptive per point: ~rel_step*scale + abs_step
    if h is not None:
        hx = hy = hz = float(h)
    # If any axis-specific step is missing, build it adaptively
    def _make_h(u, h_given):
        if h_given is not None:
            return np.asarray(h_given, dtype=float) + 0.0*u  # broadcast
        scale = np.maximum(1.0, np.abs(u))
        return rel_step * scale + abs_step
    hx = _make_h(X, hx)
    hy = _make_h(Y, hy)
    hz = _make_h(Z, hz)

    # Evaluate Φ at center once
    Phi0 = phi(X, Y, Z)

    if order == 2:
        # 2nd-order central differences: Φ''(x) ≈ [Φ(x+h)-2Φ(x)+Φ(x-h)]/h^2
        Phi_xp = phi(X + hx, Y, Z)
        Phi_xm = phi(X - hx, Y, Z)
        d2x = (Phi_xp - 2.0*Phi0 + Phi_xm) / (hx*hx)

        Phi_yp = phi(X, Y + hy, Z)
        Phi_ym = phi(X, Y - hy, Z)
        d2y = (Phi_yp - 2.0*Phi0 + Phi_ym) / (hy*hy)

        Phi_zp = phi(X, Y, Z + hz)
        Phi_zm = phi(X, Y, Z - hz)
        d2z = (Phi_zp - 2.0*Phi0 + Phi_zm) / (hz*hz)

    elif order == 4:
        # 4th-order central: Φ''(x) ≈ [ -Φ(x+2h)+16Φ(x+h)-30Φ(x)+16Φ(x-h)-Φ(x-2h) ] / (12 h^2)
        Phi_xp  = phi(X + hx,     Y, Z)
        Phi_xm  = phi(X - hx,     Y, Z)
        Phi_xpp = phi(X + 2.0*hx, Y, Z)
        Phi_xmm = phi(X - 2.0*hx, Y, Z)
        d2x = (-Phi_xpp + 16.0*Phi_xp - 30.0*Phi0 + 16.0*Phi_xm - Phi_xmm) / (12.0*hx*hx)

        Phi_yp  = phi(X, Y + hy,     Z)
        Phi_ym  = phi(X, Y - hy,     Z)
        Phi_ypp = phi(X, Y + 2.0*hy, Z)
        Phi_ymm = phi(X, Y - 2.0*hy, Z)
        d2y = (-Phi_ypp + 16.0*Phi_yp - 30.0*Phi0 + 16.0*Phi_ym - Phi_ymm) / (12.0*hy*hy)

        Phi_zp  = phi(X, Y, Z + hz)
        Phi_zm  = phi(X, Y, Z - hz)
        Phi_zpp = phi(X, Y, Z + 2.0*hz)
        Phi_zmm = phi(X, Y, Z - 2.0*hz)
        d2z = (-Phi_zpp + 16.0*Phi_zp - 30.0*Phi0 + 16.0*Phi_zm - Phi_zmm) / (12.0*hz*hz)
    else:
        raise ValueError("order must be 2 or 4")

    lap = d2x + d2y + d2z
    rho = lap / (4.0 * np.pi * G)
    return rho

    
def sph_to_cart(r: Array, th: Array, ph: Array) -> Tuple[Array, Array, Array]:
    """Convert spherical (r, θ, φ) → Cartesian (x,y,z)."""
    r = np.asarray(r, dtype=float); th = np.asarray(th, dtype=float); ph = np.asarray(ph, dtype=float)
    sinth = np.sin(th)
    x = r * sinth * np.cos(ph)
    y = r * sinth * np.sin(ph)
    z = r * np.cos(th)
    return x, y, z

def cart_to_sph(x: Array, y: Array, z: Array) -> Tuple[Array, Array, Array]:
    """Convert Cartesian (x,y,z) → spherical (r, θ, φ).


    Returns
    -------
    r : radius ≥ 0
    th : polar angle θ ∈ [0, π] (angle from +z)
    ph : azimuth φ ∈ [0, 2π) (angle from +x toward +y)
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); z = np.asarray(z, dtype=float)
    r = np.sqrt(x*x + y*y + z*z)
    # θ = arccos(z / r); define θ=0 when r=0
    th = np.where(r > 0, np.arccos(np.clip(z / r, -1.0, 1.0)), 0.0)
    # φ = atan2(y, x) mapped to [0, 2π)
    ph = np.arctan2(y, x)
    ph = np.where(ph < 0.0, ph + 2*np.pi, ph)
    return r, th, ph

def double_exponential(R, z, Sigma0=1.0, Rd=2.5, hz=0.3):
    """Volume density for a simple exponential disc: rho(R,z) = (Sigma0/(2hz)) e^{-R/Rd} e^{-|z|/hz}."""
    return (Sigma0 / (2.0 * hz)) * np.exp(-R / Rd) * np.exp(-np.abs(z) / hz)

def ferrers_bar_density(x, y, z, a=3.5, b=1.5, c=1.0, rho0=0.02, n=2, phi_bar=25.0*np.pi/180.0):
    """
    Ferrers bar: rho = rho0 (1 - m^2)^n  for m^2 = x'^2/a^2 + y'^2/b^2 + z'^2/c^2 < 1, else 0.
    The bar major axis a is rotated in the x–y plane by phi_bar (radians) from +x.
    """
    cp, sp = np.cos(phi_bar), np.sin(phi_bar)
    xp =  cp * x + sp * y
    yp = -sp * x + cp * y
    zp = z
    m2 = (xp / a) ** 2 + (yp / b) ** 2 + (zp / c) ** 2
    inside = (m2 < 1.0)
    rho = np.zeros_like(x, dtype=float)
    rho[inside] = rho0 * (1.0 - m2[inside]) ** n
    return rho

def double_expo_plus_ferrers_bar_xyz(
    x, y, z,
    # thin disc
    Sigma0_thin=1.0, Rd_thin=2.5, hz_thin=0.3,
    # thick disc
    Sigma0_thick=0.5, Rd_thick=3.0, hz_thick=0.9,
    # bar
    a_bar=4, b_bar=1.5, c_bar=0.8, rho0_bar=0.4, n_bar=2, phi_bar_deg=25.0
):
    """
    Composite test density: thin+thick exponential discs + Ferrers bar.
    Returns rho(x,y,z).
    """
    R = np.sqrt(x*x + y*y)
    phi_bar = np.deg2rad(phi_bar_deg)

    rho_thin  = double_exponential(R, z, Sigma0=Sigma0_thin,  Rd=Rd_thin,  hz=hz_thin)
    rho_thick = double_exponential(R, z, Sigma0=Sigma0_thick, Rd=Rd_thick, hz=hz_thick)
    rho_bar   = ferrers_bar_density(x, y, z, a=a_bar, b=b_bar, c=c_bar, rho0=rho0_bar, n=n_bar, phi_bar=phi_bar)

    return rho_thin + rho_thick + rho_bar

# ----------------------------- Examples -------------------------------
if __name__ == "__main__":
    # Simple demo density: sum of a Plummer sphere + weak m=2 bar-like term
    M, a = 1.0, 1.0
    # def rho_demo_sph(r, th, ph):
    #     return (3*M/(4*np.pi*a**3)) * (1 + (r**2)/(a**2))**(-2.5) * (1 + 0.5*np.cos(2*ph)*np.sin(th)**2)

    def rho_gt(x, y, z):
        return double_expo_plus_ferrers_bar_xyz(x, y, z)

    mp = MultipoleExpansion(rho_gt, lmax=12, mmax=8, NR=50, rmin=1e-3, rmax=50.0)
    # r = np.geomspace(1e-3, 20.0, 200)
    # phi_line = mp.potential(r, np.pi/2, 0.0)
    # G = 1
    # phi_ana = -G*M/np.sqrt(r**2 + a**2)
    # rel_err = np.mean(np.abs((phi_line - phi_ana)/phi_ana))

    print("Multipole demo: Φ(r_eq) computed.",)# "Max rel error vs. analytic:", rel_err

    import matplotlib.pyplot as plt

    # # Make a grid in (x, y, z=0)
    # nxy = 100
    # x = np.linspace(-5, 5, nxy)
    # y = np.linspace(-5, 5, nxy)
    # X, Y = np.meshgrid(x, y)
    # Rxy = np.sqrt(X**2 + Y**2)
    # THxy = np.arccos(np.clip(0.0 / np.sqrt(Rxy**2 + 0.0**2), -1, 1))  # z=0, so θ=π/2 except at r=0
    # PHxy = np.arctan2(Y, X)

    # # Evaluate reconstructed and analytic potentials
    # Phi_rec = mp.potential(Rxy, THxy, PHxy)
    # Phi_ana = -G*M/np.sqrt(Rxy**2 + a**2)


    # # Compute residuals
    # residual = Phi_rec - Phi_ana

    # # Plot side by side
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # im0 = axs[0].imshow(Phi_rec, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
    # axs[0].set_title('Reconstructed Potential')
    # plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # im1 = axs[1].imshow(Phi_ana, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
    # axs[1].set_title('Analytic Potential')
    # plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # im2 = axs[2].imshow(residual/Phi_ana, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='coolwarm',
    #                     vmin = -np.max(np.abs(residual/Phi_ana)), vmax = np.max(np.abs(residual/Phi_ana)))
    # axs[2].set_title('Residual (Recon - Analytic)')
    # plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # for ax in axs:
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')

    # plt.tight_layout()
    # plt.show()

    # --- Projected surface density maps: face-on (z=0) and side-on (y=0) ---

    # Face-on: integrate along z, grid in (x, y)
    nx = 100
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, nx)
    X, Y = np.meshgrid(x, y)
    Z = np.linspace(-10, 10, nx)
    dZ = Z[1] - Z[0]

    # Ground truth surface density (face-on)
    Sigma_gt_faceon = np.zeros_like(X)
    for i in tqdm(range(nx)):
        z_slice = Z[i]
        Sigma_gt_faceon += rho_gt(X, Y, np.full_like(X, z_slice)) * dZ

    # Fitted surface density (face-on)
    Sigma_fit_faceon = np.zeros_like(X)
    for i in tqdm(range(nx)):
        z_slice = Z[i]
        r, th, ph = cart_to_sph(X, Y, np.full_like(X, z_slice))
        Sigma_fit_faceon += mp.density(r, th, ph) * dZ


    # Residual (face-on)
    Sigma_res_faceon = Sigma_fit_faceon - Sigma_gt_faceon

    # Side-on: integrate along x, grid in (y, z)
    Y2 = np.linspace(-10, 10, nx)
    Z2 = np.linspace(-10, 10, nx)
    Yg, Zg = np.meshgrid(Y2, Z2)
    X2 = np.linspace(-10, 10, nx)
    dX = X2[1] - X2[0]

    # Ground truth surface density (side-on)
    Sigma_gt_sideon = np.zeros_like(Yg)
    for i in tqdm(range(nx)):
        x_slice = X2[i]
        Sigma_gt_sideon += rho_gt(np.full_like(Yg, x_slice), Yg, Zg) * dX

    # Fitted surface density (side-on)
    Sigma_fit_sideon = np.zeros_like(Yg)
    for i in tqdm(range(nx)):
        x_slice = X2[i]
        r, th, ph = cart_to_sph(np.full_like(Yg, x_slice), Yg, Zg)
        Sigma_fit_sideon += mp.density(r, th, ph) * dX


    # Residual (side-on)
    Sigma_res_sideon = Sigma_fit_sideon - Sigma_gt_sideon

    # --- Plotting ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Face-on
    im0 = axs[0, 0].imshow(Sigma_gt_faceon, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
    axs[0, 0].set_title('Face-on: Ground Truth Σ')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(Sigma_fit_faceon, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
    axs[0, 1].set_title('Face-on: Fitted Σ')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[0, 2].imshow(Sigma_res_faceon/Sigma_gt_faceon, origin='lower', 
    extent=[x.min(), x.max(), y.min(), y.max()], cmap='coolwarm', vmin = -0.5, vmax = 0.5)
    axs[0, 2].set_title('Face-on: Residual (Fit - Truth)')
    plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    for ax in axs[0, :]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Side-on
    im3 = axs[1, 0].imshow(Sigma_gt_sideon, origin='lower', extent=[Y2.min(), Y2.max(), Z2.min(), Z2.max()], cmap='viridis')
    axs[1, 0].set_title('Side-on: Ground Truth Σ')
    plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im4 = axs[1, 1].imshow(Sigma_fit_sideon, origin='lower', extent=[Y2.min(), Y2.max(), Z2.min(), Z2.max()], cmap='viridis')
    axs[1, 1].set_title('Side-on: Fitted Σ')
    plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im5 = axs[1, 2].imshow(Sigma_res_sideon/Sigma_gt_sideon, origin='lower', 
    extent=[Y2.min(), Y2.max(), Z2.min(), Z2.max()], cmap='coolwarm', vmin = -1, vmax = 1)
    axs[1, 2].set_title('Side-on: Residual (Fit - Truth)')
    plt.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

    for ax in axs[1, :]:
        ax.set_xlabel('y')
        ax.set_ylabel('z')

    plt.tight_layout()
    fig.savefig("//Users/hanyuan/Desktop/PhD_projects/Schwarchild_bar/multipole_surface_density_maps.png", dpi=150)
    plt.show()
