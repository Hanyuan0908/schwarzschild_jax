import agama
agama.setUnits(mass=1, length=1, velocity=1)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from potential_cylspline import *


def mn_phi_cyl(R, z, M, a, b, G):
    """
    Miyamoto–Nagai potential Φ(R,z) = - G M / sqrt(R^2 + (a + sqrt(z^2 + b^2))^2)
    """
    R = np.asarray(R, float); z = np.asarray(z, float)
    beta = np.sqrt(z*z + b*b)
    D = np.sqrt(R*R + (a + beta)**2)
    return -G * M / D

def mn_rho_cyl(R, z, M, a, b):
    """
    Miyamoto–Nagai density ρ(R,z) from the analytic potential–density pair.
    ρ = (b^2 M / 4π) * [ a R^2 + (a + 3β)(a + β)^2 ] / [ β^3 * (R^2 + (a + β)^2)^(5/2) ],
    where β = sqrt(z^2 + b^2).
    """
    R = np.asarray(R, float); z = np.asarray(z, float)
    beta = np.sqrt(z*z + b*b)
    D2 = R*R + (a + beta)**2
    num = a * R*R + (a + 3.0*beta) * (a + beta)**2
    den = (beta**3) * (D2**2.5)
    return (b*b * M / (4.0 * np.pi)) * (num / den)

def mn_phi_cart(x, y, z, M=1e9, a=3, b=0.8, G=4.3e-6):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    R = np.sqrt(x*x + y*y)
    return mn_phi_cyl(R, z, M=M, a=a, b=b, G=G)

def mn_rho_cart(x, y, z, M=1e9, a=3, b=0.8):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    R = np.sqrt(x*x + y*y)
    return mn_rho_cyl(R, z, M=M, a=a, b=b)


if __name__ == "__main__":
    # Simple demo density: sum of a Plummer sphere + weak m=2 bar-like term
    M, a = 1.0, 1.0
    # def rho_demo_sph(r, th, ph):
    #     return (3*M/(4*np.pi*a**3)) * (1 + (r**2)/(a**2))**(-2.5) * (1 + 0.5*np.cos(2*ph)*np.sin(th)**2)

    def rho_gt(x, y, z):
        return mn_rho_cart(x, y, z)
    
    def Phi_gt(pts):
        x, y, z = pts[:,0], pts[:,1], pts[:,2]
        return mn_phi_cart(x, y, z)

    cs = CylSpline(rho_gt, NR = 50, Nz_pos = 30, mmax = 0, Rmin = 1e-2, Rmax = 30.0, Zmin = 1e-2, Zmax = 20.0,)
    cs.compute_rho_m()
    # 3a) compute Φ_m with fixed tensor grid (parallel/JAX-friendly)
    # cs.compute_phi_m_grid_fixed(N_int=50000, rule="simpson")
    # cs.compute_phi_m_grid_fixed(N_int=10000, rule="simpson")
    # cs.compute_phi_m_grid_fixed(N_int=10000, rule="trapezoid")
    cs.compute_phi_m_grid_fixed_mapped(N_int=10000)

    # 3b) or compute Φ_m with adaptive integrator (eval budget + tol)
    # cs.compute_phi_m_grid_adaptive(tol=1e-4, max_evals=10_000)
    print("Cylindrical demo: Φ(r_eq) computed.",)# "Max rel error vs. analytic:", rel_err

    def rho_input(x):
        return rho_gt(x[:,0], x[:,1], x[:,2])
    pot_agama = agama.Potential(type='Cylspline', density=rho_input, gridSizeR=50, gridSizez=30, rmin=1e-3, rmax=30,zmin=1e-3, zmax=20,
                        symmetry = 'none', mmax=7)

    # --- Projected surface density maps: face-on (z=0) and side-on (y=0) ---

    # Face-on: integrate along z, grid in (x, y)
    nx = 100
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, nx)
    X, Y = np.meshgrid(x, y)
    Z = np.linspace(-5, 5, nx)
    dZ = Z[1] - Z[0]

    # Ground truth surface density (face-on)
    Sigma_gt_faceon = np.zeros_like(X)
    for i in tqdm(range(len(Z))):
        z_slice = Z[i]
        Sigma_gt_faceon += rho_gt(X, Y, np.full_like(X, z_slice)) * dZ

    # Fitted surface density (face-on)
    Sigma_fit_faceon = np.zeros_like(X)
    for i in tqdm(range(len(Z))):
        z_slice = Z[i]
        # r, phi, z = cartesian_to_cylindrical(X, Y, np.full_like(X, z_slice))
        Sigma_fit_faceon += cs.density_cartesian(X, Y, np.full_like(X, z_slice)) * dZ
        

    # Residual (face-on)
    Sigma_res_faceon = Sigma_fit_faceon - Sigma_gt_faceon

    # Side-on: integrate along x, grid in (y, z)
    Y2 = np.linspace(-10, 10, nx)
    Z2 = np.linspace(-10, 10, nx)
    Yg, Zg = np.meshgrid(Y2, Z2)
    X2 = np.linspace(-10, 10, nx)
    # X2 = np.array([-0.1,0.1])#np.linspace(-10, 10, nx)
    dX = X2[1] - X2[0]

    # Ground truth surface density (side-on)
    Sigma_gt_sideon = np.zeros_like(Yg)
    for i in tqdm(range(len(X2))):
        x_slice = X2[i]
        Sigma_gt_sideon += rho_gt(np.full_like(Yg, x_slice), Yg, Zg) * dX

    # Fitted surface density (side-on)
    Sigma_fit_sideon = np.zeros_like(Yg)
    for i in tqdm(range(len(X2))):
        x_slice = X2[i]
        # r, phi, z = cartesian_to_cylindrical()
        Sigma_fit_sideon += cs.density_cartesian(np.full_like(Yg, x_slice), Yg, Zg) * dX


    # Residual (side-on)
    Sigma_res_sideon = Sigma_fit_sideon - Sigma_gt_sideon

    # --- Plotting ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Face-on
    im0 = axs[0, 0].imshow(Sigma_gt_faceon, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis',
    norm = mpl.colors.LogNorm())
    axs[0, 0].set_title('Face-on: Ground Truth Σ')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(Sigma_fit_faceon, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis',
    norm = mpl.colors.LogNorm())
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
    im3 = axs[1, 0].imshow(Sigma_gt_sideon, origin='lower', extent=[Y2.min(), Y2.max(), Z2.min(), Z2.max()], cmap='viridis',
    norm = mpl.colors.LogNorm())
    axs[1, 0].set_title('Side-on: Ground Truth Σ')
    plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im4 = axs[1, 1].imshow(Sigma_fit_sideon, origin='lower', extent=[Y2.min(), Y2.max(), Z2.min(), Z2.max()], cmap='viridis',
    norm = mpl.colors.LogNorm())
    axs[1, 1].set_title('Side-on: Fitted Σ')
    plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im5 = axs[1, 2].imshow(Sigma_res_sideon/Sigma_gt_sideon, origin='lower', 
    extent=[Y2.min(), Y2.max(), Z2.min(), Z2.max()], cmap='coolwarm', vmin = -0.5, vmax = 0.5)
    axs[1, 2].set_title('Side-on: Residual (Fit - Truth)')
    plt.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

    for ax in axs[1, :]:
        ax.set_xlabel('y')
        ax.set_ylabel('z')

    plt.tight_layout()
    fig.savefig("//Users/hanyuan/Desktop/PhD_projects/Schwarchild_bar/cylspline_MNdisc_surface_density_maps.png", dpi=150)
    # plt.show()

    Potential_agama_faceon = np.zeros_like(X)
    for i in tqdm(range(len(Z))):

        z_slice = Z[i]
        # r, th, ph = cart_to_sph(np.full_like(Yg, x_slice), Yg, Zg)
        points = np.column_stack((X.ravel(), Y.ravel(), np.full_like(X.ravel(), z_slice)))
        Potential_agama_faceon += (Phi_gt(points) * dZ).reshape(X.shape)

    Potential_fit_faceon = np.zeros_like(X)
    for i in tqdm(range(len(Z))):
        z_slice = Z[i]
        Potential_fit_faceon += (cs.potential_cartesian(X, Y, np.full_like(X, z_slice)) * dZ)
    # for i in tqdm(range(len(Z))):
    #     z_slice = Z[i]
    #     points = np.column_stack((X.ravel(), Y.ravel(), np.full_like(X.ravel(), z_slice)))
    #     Potential_fit_faceon += (pot_agama.potential(points) * dZ).reshape(X.shape)


    Potential_agama_sideon = np.zeros_like(Yg)
    for i in tqdm(range(len(X2))):
        x_slice = X2[i]
        # r, th, ph = cart_to_sph(np.full_like(Yg, x_slice), Yg, Zg)
        points = np.column_stack((np.full_like(Yg.ravel(), x_slice), Yg.ravel(), Zg.ravel()))
        Potential_agama_sideon += (Phi_gt(points) * dX).reshape(Yg.shape)

    Potential_fit_sideon = np.zeros_like(Yg)
    for i in tqdm(range(len(X2))):
        x_slice = X2[i]
        Potential_fit_sideon += (cs.potential_cartesian(np.full_like(Yg, x_slice), Yg, Zg) * dX)
    # for i in tqdm(range(nx)):
    #     x_slice = X2[i]
    #     points = np.column_stack((np.full_like(Yg.ravel(), x_slice), Yg.ravel(), Zg.ravel()))
    #     Potential_fit_sideon += (pot_agama.potential(points) * dX).reshape(Yg.shape)

    # Residual (side-on)
    Potential_residual_faceon = Potential_fit_faceon - Potential_agama_faceon
    Potential_residual_sideon = Potential_fit_sideon - Potential_agama_sideon

    # --- Plotting ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    vmin = np.min([-Potential_agama_faceon.min(), -Potential_fit_faceon.min()])
    vmax = np.max([-Potential_agama_faceon.max(), -Potential_fit_faceon.max()])
    # Face-on
    im0 = axs[0, 0].imshow(-Potential_agama_faceon, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis',
    vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Face-on: Ground truth potential')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(-Potential_fit_faceon, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis',
    vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Face-on: Fitted Potential')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[0, 2].imshow(Potential_residual_faceon/Potential_agama_faceon, origin='lower', 
    extent=[x.min(), x.max(), y.min(), y.max()], cmap='coolwarm', vmin=-0.1, vmax=0.1)
    axs[0, 2].set_title('Face-on: Residual (Fit - Truth)')
    plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    for ax in axs[0, :]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Side-on
    vmin = np.min([-Potential_agama_sideon.min(), -Potential_fit_sideon.min()])
    vmax = np.max([-Potential_agama_sideon.max(), -Potential_fit_sideon.max()])
    im3 = axs[1, 0].imshow(-Potential_agama_sideon, origin='lower', extent=[Y2.min(), Y2.max(), Z2.min(), Z2.max()], cmap='viridis',
    vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Side-on: Ground truth Potential')
    plt.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im4 = axs[1, 1].imshow(-Potential_fit_sideon, origin='lower', extent=[Y2.min(), Y2.max(), Z2.min(), Z2.max()], cmap='viridis',
    vmin=vmin, vmax=vmax)
    axs[1, 1].set_title('Side-on: Fitted Potential')
    plt.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)

    im5 = axs[1, 2].imshow(Potential_residual_sideon/Potential_agama_sideon, origin='lower',
    extent=[Y2.min(), Y2.max(), Z2.min(), Z2.max()], cmap='coolwarm', vmin=-0.1, vmax=0.1)
    axs[1, 2].set_title('Side-on: Residual (Fit - Truth)')
    plt.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)

    for ax in axs[1, :]:
        ax.set_xlabel('y')
        ax.set_ylabel('z')

    plt.tight_layout()
    fig.savefig("//Users/hanyuan/Desktop/PhD_projects/Schwarchild_bar/cylspline_MNdisc_projected_potentials_maps.png", dpi=150)
    plt.show()

    plt.figure()
    plt.hist((Potential_agama_faceon/Potential_fit_faceon).ravel(), bins = 100)
    plt.hist((Potential_agama_sideon/Potential_fit_sideon).ravel(), bins = 100)
    plt.show()