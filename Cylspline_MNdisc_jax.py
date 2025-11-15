import jax
import jax.numpy as jnp
import pickle
from constants import *
from cubic_spline import *
from functools import partial


def mn_rho_cyl(R, z, M, a, b):
    """
    Miyamoto–Nagai density ρ(R,z) from the analytic potential–density pair.
    ρ = (b^2 M / 4π) * [ a R^2 + (a + 3β)(a + β)^2 ] / [ β^3 * (R^2 + (a + β)^2)^(5/2) ],
    where β = sqrt(z^2 + b^2).
    """
    beta = jnp.sqrt(z*z + b*b)
    D2 = R*R + (a + beta)**2
    num = a * R*R + (a + 3.0*beta) * (a + beta)**2
    den = (beta**3) * (D2**2.5)
    return (b*b * M / (4.0 * jnp.pi)) * (num / den)

@jax.jit
def rho_xyz(x,y,z):
    R = jnp.sqrt(x*x + y*y)
    return mn_rho_cyl(R, z, M=1e9, a=3, b=0.8)

# ============= Change the density rho_xyz above to your desired density profile ============= #

@jax.jit
def cylindrical_to_cartesian(R, phi, z):
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    return x, y, z

@jax.jit
def rho_Rzphi(R, z, phi):
    z = jnp.abs(z)  # even symmetry
    x, y, zz = cylindrical_to_cartesian(R, phi, z)
    return rho_xyz(x, y, zz)

@jax.jit
def rho_last(R, z, m, phi, Nphi=200):
    dphi = (2*jnp.pi) / Nphi
    vals = rho_Rzphi(R, z, phi)
    exp_ph = jnp.exp(-1j * m * phi)
    rho_m_stack = vals * exp_ph * dphi / (2.0 * jnp.pi)
    return rho_m_stack

@jax.jit
def rho_phiZRm(R, z, m, phi):
    return jnp.sum(jax.vmap(rho_last, in_axes=(None, None, None, 0))(R, z, m, phi), axis=0)

@jax.jit
def compute_rho_m(R, z, m, phi):
    return jax.vmap(rho_phiZRm, in_axes=(None, None, 0, None))(R, z, m, phi)

@jax.jit
def jax_rho_m_eval(m, R, z, Rgrid, Zgrid, rho_real, rho_img, Mx_real, My_real, Mx_img, My_img):
    real_values = rho_real[m]
    imag_values = rho_img[m]
    M_x_real = Mx_real[m]
    M_y_real = My_real[m]
    M_x_imag = Mx_img[m]
    M_y_imag = My_img[m]

    shape = R.shape
    pts = jnp.column_stack((R.ravel(), jnp.abs(z).ravel()))

    real_part = cubic_spline_evaluate(pts, (Rgrid, Zgrid), real_values, M_x_real, M_y_real, fill_value=0.0).reshape(shape)
    imag_part = cubic_spline_evaluate(pts, (Rgrid, Zgrid), imag_values, M_x_imag, M_y_imag, fill_value=0.0).reshape(shape)

    return real_part + 1j * imag_part



@jax.jit
def jax_hypergeom_m(m, x):
    """
    m: int,
    x: array-like
    """


    y = 1.0 - x
    y2 = y*y
    z = jnp.log(jnp.where(y > 1e-12, y, 1e-12))

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

    F = jnp.where(x < X_THRESHOLD1[m],
                 jnp.where(x < X_THRESHOLD0[m], val_1, val_2),
                 val3)

    return F

@jax.jit
def jax_legendreQ(n, x):
    """
    n: float,
    x: array-like
    """

    x = jnp.where(x < 1.0, 1.0, x)
    out = jnp.empty_like(x)
    m = jnp.round(n + 0.5).astype(jnp.int32)

    pref = Q_PREFACTOR[m] / jnp.sqrt(x) / (x**m)
    F = jax_hypergeom_m(m, 1.0/(x*x))
    out = pref * F

    return out

@jax.jit
def jax_kernel_Xi_m(m, R, z, Rp, zp):

    """
    m: int,
    R: float,
    z: float,
    Rp: array-like,
    zp: array-like
    """
    zeros = jnp.zeros_like(Rp, dtype=float)

    val1 = zeros
    val2 = 1.0 / jnp.sqrt(R*R + Rp*Rp + (z - zp)**2)

    val_zero = jax.lax.cond(m>0, lambda: val1, lambda: val2)
    
    Rp_reg = Rp
    dz = (z - zp)
    chi = (R*R + Rp_reg*Rp_reg + dz*dz) / (2.0 * R * Rp_reg)
    chi = jnp.maximum(chi, 1.0)
    Q = jax_legendreQ(m - 0.5, chi)
    val_nonzero = (1.0 / (jnp.pi * jnp.sqrt(R * Rp_reg))) * Q


    val_out = jnp.where(Rp<1e-3, val_zero, val_nonzero)

    out = jax.lax.cond(R < 1e-3, lambda: val_zero, lambda: val_out)

    return out


@partial(jax.jit, static_argnames=['n'])
def simpson_weights(n):
    w = jnp.ones(n)
    w = w.at[1:-1:2].set(4.0)
    w = w.at[2:-1:2].set(2.0)
    w *= (1.0 / (n - 1)) / 3.0   # h = 1/(n-1), scale by h/3
    return w


@jax.jit
def _xieta_to_Rz_jacobian(xi, eta, Rzminmax):
    # Rmin_map = R[1]
    # Rmax_map = Rmax
    # zmin_map = Z_nonneg[1]
    # zmax_map = Zmax
    Rmin_map = Rzminmax[0]
    Rmax_map = Rzminmax[1]
    zmin_map = Rzminmax[2]
    zmax_map = Rzminmax[3]

    # Precompute logs
    LR = jnp.log(1.0 + Rmax_map / Rmin_map)
    LZ = jnp.log(1.0 + zmax_map / zmin_map)

    # Map to physical coordinates
    pR = jnp.power(1.0 + Rmax_map / Rmin_map, xi)
    pZ = jnp.power(1.0 + zmax_map / zmin_map, eta)
    Rp = Rmin_map * (pR - 1.0)
    zp = zmin_map * (pZ - 1.0)

    # Jacobian part from the coordinate transform (no 2πR' here)
    dR_dxi  = LR * (Rmin_map + Rp)
    dz_deta = LZ * (zmin_map + zp)
    J = dR_dxi * dz_deta
    return Rp, zp, J

@jax.jit
def m_wrapper(m, R0, z0, Rp, zp):
    rho_grid = jax_rho_m_eval(m.astype(int), Rp, zp, R, Z_nonneg, rho_real, rho_img, Mx_real, My_real, Mx_img, My_img)

    return jax.vmap(R_wrapper, in_axes=(None, 0, None, None, None, None))(m, R0, z0, Rp, zp, rho_grid)

@jax.jit
def R_wrapper(m, R0, z0, Rp, zp, rho_grid):
    return jax.vmap(Z_wrapper, in_axes=(None, None, 0, None, None, None))(m, R0, z0, Rp, zp, rho_grid)

@jax.jit
def Z_wrapper(m, R0, z0, Rp, zp, rho_grid):
    Xi_plus  = jax_kernel_Xi_m(m, R0, z0, Rp, zp)
    Xi_minus = jax_kernel_Xi_m(m, R0, z0, Rp, -zp)
    Xi_sum   = Xi_plus + Xi_minus

    F = rho_grid * Xi_sum * (2.0 * np.pi) * Rp * Jmap

    I = np.sum(W2D * F)

    return -G * I

NR, NZ, Rmin, Rmax, Zmin, Zmax, Mmax = 50, 30, 1e-2, 30.0, 1e-2, 20.0, 0

Nphi = 200

M = jnp.arange(0, Mmax + 1)

R = jnp.geomspace(jnp.maximum(Rmin, 1e-3), Rmax, NR)
R0_eff = R[NR // 2]

Zpos = jnp.geomspace(jnp.maximum(Zmin, 1e-3), Zmax, NZ)
Z_nonneg = jnp.concatenate([jnp.array([0.0]), Zpos])

Rg, Zg = jnp.meshgrid(R, Z_nonneg, indexing="ij")
phi = jnp.linspace(0.0, 2*jnp.pi, Nphi, endpoint=False)

dphi = (2*jnp.pi) / Nphi

rho_m = jax.vmap(compute_rho_m, in_axes=(0, 0, None, None))(Rg, Zg, M, phi).transpose(1,0,2) 


rho_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
rho_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
Mx_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
My_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
Mx_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
My_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
for m in M.astype(int):
    rho_real = rho_real.at[m].set(rho_m[m].real)
    M_x, M_y = jax_precompute_splines((R, Z_nonneg), rho_m[m].real)
    Mx_real = Mx_real.at[m].set(M_x)
    My_real = My_real.at[m].set(M_y)
    rho_img = rho_img.at[m].set(rho_m[m].imag)
    M_x, M_y = jax_precompute_splines((R, Z_nonneg), rho_m[m].imag)
    Mx_img = Mx_img.at[m].set(M_x)
    My_img = My_img.at[m].set(M_y)


N_int = 10_000
base = jnp.maximum(9, jnp.sqrt(jnp.maximum(16, N_int)).astype(int))
base += jnp.abs(base % 2 - 1)  # make it odd

n_xi = base
n_eta = base
wxi  = simpson_weights(int(n_xi))
weta = simpson_weights(int(n_eta))

xi  = jnp.linspace(0.0, 1.0, n_xi)
eta = jnp.linspace(0.0, 1.0, n_eta)
XI, ETA = jnp.meshgrid(xi, eta, indexing="ij")
Rp, zp, Jmap = _xieta_to_Rz_jacobian(XI, ETA, jnp.array([R[1], Rmax, Z_nonneg[1], Zmax])) 
W2D = jnp.einsum('i,j->ij', wxi, weta)

phi_m = jax.vmap(m_wrapper, in_axes=(0, None, None, None, None))(M.astype(int), R, Z_nonneg, Rp, zp)

phi_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
phi_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
phi_Mx_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
phi_My_real = jnp.zeros((len(M), len(R), len(Z_nonneg)))
phi_Mx_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
phi_My_img = jnp.zeros((len(M), len(R), len(Z_nonneg)))
for m in M.astype(int):
    phi_real = phi_real.at[m].set(phi_m[m].real)
    M_x, M_y = jax_precompute_splines((R, Z_nonneg), phi_m[m].real)
    phi_Mx_real = phi_Mx_real.at[m].set(M_x)
    phi_My_real = phi_My_real.at[m].set(M_y)
    phi_img = phi_img.at[m].set(phi_m[m].imag)
    M_x, M_y = jax_precompute_splines((R, Z_nonneg), phi_m[m].imag)
    phi_Mx_img = phi_Mx_img.at[m].set(M_x)
    phi_My_img = phi_My_img.at[m].set(M_y)

Phi_m_grid = {
    'Rgrid': R,
    'Zgrid': Z_nonneg,
    'm': M,
    'Phi_m_real': phi_real,
    'Phi_m_img': phi_img,
    'Mx_real': phi_Mx_real,
    'My_real': phi_My_real,
    'Mx_img': phi_Mx_img,
    'My_img': phi_My_img,
}

with open('./pot_data/MN_disc.pkl', 'wb') as f:
    pickle.dump(Phi_m_grid, f)