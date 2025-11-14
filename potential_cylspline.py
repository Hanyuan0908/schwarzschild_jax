import numpy as np
from tqdm import tqdm
from typing import Callable, Optional, Tuple, Dict, Union, List
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gamma, hyp2f1

Array = np.ndarray
ArrayLike = Union[float, np.ndarray]

import jax
import jax.numpy as jnp
from functools import partial
from constants import EPSILON
# -------------------- coordinate helpers --------------------
@jax.jit
def cartesian_to_cylindrical(x,y,z):
    R   = jnp.sqrt(x*x + y*y)
    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi < 0.0, phi + 2*jnp.pi, phi)
    phi = jnp.where(R == 0.0, 0.0, phi)  # define φ=0 on axis
    return R, phi, z

@jax.jit
def cylindrical_to_cartesian(R, phi, z):
    x = R * jnp.cos(phi); y = R * jnp.sin(phi)
    return x, y, z

# -------------------- AGAMA-style Legendre Q (Padé + asymptotic) --------------------
# Tables from the AGAMA C++ snippet you provided (m = 0..12).
MMAX_HYPERGEOM = 12
HYPERGEOM_0 = np.array([
    [ 0.4529874071816636,-2.692539697590362,-6.480598406507461,-3.860216328610213,-2.609146033646615,
     -0.1909432553353492,-1.460299624021841,-0.01541563568408259,-1.115369096436195],
    [ 0.08938502676040028,-2.525406068615592,-3.802041706399501,-2.691157667804647,-2.741391856196645,
     -0.1823978151361306,-1.467764617695739,-0.01514303374682610,-1.116215391130251],
    [-0.01069787527041158,-1.537005620596391,-1.697345768635404,-0.3378529865322015,-1.983063657443297,
     -0.09488668216801367,-1.362931464322307,-0.00940689438170889,-1.093693723592968],
    [-0.01711490718455377,-0.6009550456991466, 3.394854752517220,36.72266736665577,-9.201204857656629,
      0.01516766397285159,-1.228118125014300,-0.005976086542472318,-1.080099304233194],
    [-0.02749247751103787,-0.3072229139612030, 5.687071537359105,55.96560640229840,-9.348637781981967,
      0.0007050923061239599,-1.068784792886700, 0.0006587322854650128, 0.1314557788692205],
    [ 0.006592755012603758, 0.1271086283761586,-20.85294538326066,368.2891340792655,17.55960611497124,
      0.01287836941499270,-2.119217276127404, 0.005841336094937430,-1.041259452130866],
    [ 0.02415357420175733, 0.2760551779728967, -8.304528092063684, 49.00778641555581, 5.706942819375404,
      0.0000135380497674,-1.077581157358874, 0.01898377989044321,-0.8802786435315881],
    [-0.4755438570635217,-15.23176265534132,-26.14525965096805, 22.72480434937186,-3.726962896593618,
     12.26068097538548, 2.374616974037387,-0.00001072561873862000,-0.9939982148579712],
    [-0.01778950682637795,-0.1621708071527029, -6.722812444954199, -3.605415274607454, 42.56877984182697,
    1763.410384466364, -40.89722407762218, 4.04152404e-8,-0.9643299106460461],
    [-0.01640040314046018,-0.1567441655078544, -6.696210685394554, 5.878476915868881,-15.59083749781898,
     297.2840667256496, 18.02878411264815,-1.6468589e-9,-0.9155934138921334],
    [-0.04537030719843308,-0.3438471580930914, -5.110748703261416, 4.979640874599663, -4.697175985956599,
      33.93946493242602, 5.914298150113606, 9.7752486955e-9, 0.01489956391058121],
    [ 0.2560046720818122, 3.738379858847606,-10.25803052150991,11.73970408392119,-1.965832605102681,
      0.3053195680701161,-35.13200601587556,1238.732680559852,35.14765318693473],
    [ 0.01532856961552287, 0.1296876459958492, -5.869778335825212, 4.144607847312670, -1.999697044584623,
     -29.36865667344516, 5.786781200597918, 63.58059663540046,-3.806468298143671],
])
HYPERGEOM_I = np.array([
    [ 0.6265483823527411,-0.8488996006421011,-2.430767712870059,-0.2216645603123820,-1.377915333154397,
     -0.01067779281977426,-1.106258117483859,-0.0008017018734870802,-1.025715949003203],
    [ 0.2556745290319481,-1.302202831061045,-1.856484357772605,-0.1506639551665747,-1.365766354550209,
     -0.008801806475184472,-1.097860353907581,-0.0006463098611572502,-1.023023211337991],
    [ 0.06793076058208223,-1.259938656832443,-1.383753616440758,-0.04811851897121551,-1.252578275690214,
     -0.004438147879886131,-1.071030099270125,-0.0003371916228163400,-1.016826439397124],
    [-0.1438872110578193,-1.318526106712988,-1.165709435421967,-0.01078909626071359,-1.140430116301888,
     -0.001811274363462059,-1.048091679713478,-0.0001577543940147300,-1.011553061701041],
    [-0.5623160680982770,-1.576021970674381,-1.076471896362168,-0.001780342522184680,-1.062427091799673,
      0.004018205261775791,-0.6514030690056149, 0.005561736279330680,-1.027642872383092],
    [-9.162872732083632, 42.14305648246658, 5.624924575599729, 2.272916747830027,-1.389775678893547,
     -0.00073079680223201,-1.034210205811073,-0.00008747485028171,-1.0085136703608],
    [ 0.2017739963700901, 0.5462246106199197, -5.109468125375064,14.41419452992633, 2.502624381778485,
     -0.0000680997442925,-1.030883958557467,-0.00008191929998314,-1.008456613288245],
    [ 0.3185170028368671, 0.8886260784862031, -3.580049820836553, 5.651162454123797, 1.186628617227599,
     -0.00003820523068188,-1.023368519956452,-0.00004924296499383,-1.006649542714069],
    [ 0.5045523625514224, 1.485758862452970, -2.752242779713227, 2.595936432732558, 0.4785798705675604,
     -0.00002433498096823,-1.019227289991061,-0.00003585972755995,-1.005789911366520],
    [ 1.398160625476730, 2.954532545810714, -2.136522020336696, 1.061026744232020, -0.06909753819383704,
     -0.00001821885545569,-1.015887980712423,-0.00002582418432982,-1.004995632391522],
    [ 3.620215546579624, 5.686835654840336, -1.799598166133843, 0.5134745736144935, -0.3602442057709179,
     -0.00001410577594344,-1.013913058391083,-0.00002176004393978,-1.004725750313438],
    [ 8.776955057607215,10.69299243000038, -1.596330222811836, 0.2810073271616607, -0.5309379037693800,
     -0.00001164458682589,-1.013280572890286,-0.00002307480563413,-1.005031422945444],
    [20.02358305143908,19.69122956965990, -1.465053947210626, 0.1693355984048097, -0.6378402328091336,
     -0.00001088292906364,-1.014557343074211,-0.00003173874282459,-1.005763792645454],
])
HYPERGEOM_1 = np.array([
    [  0.9360775742346216,-0.2250790790392765, 0.0348401207694437,-0.0422023273198643, 0.0104808433089421,
     -0.0230793977530508, 0.0049617494377955,-0.0158670859552224, 0.0028807359141383,-0.0120862568799546],
    [ 0.1430450323100617,-0.9003163161571056,0.02156517827104491,-0.8440465463972865,0.008039995073102123,
     -0.8308583191098289,0.004137595481807615,-0.8250884696715662,0.002509932082129108,-0.8218654678369117],
    [-2.819671260176213,-2.400843509752283,-2.866871055726074,-5.251845177583119,-2.875549627231282,
     -8.123948009073887,-2.878479952408139,-11.00117959562089,-2.879798366033066,-13.88039456791229],
    [-11.3768305631473,-5.76202442340548,-22.46842570169623,-22.68797116715908,-33.54140985134508,
     -50.69343557662106,-44.61066123524255,-89.76962550026647,-55.67872510000488,-139.914377244556],
    [-33.53009359531525,-13.17034153921252,-110.3361852693207,-81.49148827387747,-230.4962906372769,
     -248.2943783344704,-394.0183070924650,-556.9380847363468,-600.9038975602715,-1050.785527061154],
    [-87.51906383078700,-29.26742564269449,-434.6509534806616,-261.5776166815820,-1212.744309671997,
     -1042.223316465678,-2593.120194540037,-2887.827106040317,-4747.096184506519,-6486.330413957741],
    [-214.1711215952598,-63.85620140224254,-1500.709045078266,-778.2474545898310,-5397.230112323440,
     -3927.717622383053,-14120.94007905760,-13174.21952507649,-30568.64501900970,-34736.71163838528],
    [-503.6105492173122,-137.5364337894455,-4742.610771427904,-2191.986913519288,-21381.51289794725,
     -13665.66841397181,-67219.94462809997,-54567.77318079023,-169894.9790469529,-166900.6500021825],
    [-1152.612120663817,-293.4110587508170,-14062.58521759393,-5923.235748532118,-77715.55133278912,
     -44701.91978970333,-289167.5872258646,-209540.2490142344,-843996.0107326956,-735846.4213429555],
    [-2587.023565073958,-621.3410655899655,-39737.92516363194,-15494.69282314976,-264294.3690284689,
     -139210.1308329862,-1149012.999729593,-756955.0864043624,-3836087.008386949,-3024863.489811182],
    [-5721.752022023828,-1308.086453873611,-108138.6205048347,-39487.85982630963,-852386.5359668258,
     -416473.5216056094,-4282240.812990009,-2600067.332801687,-16217451.44075212,-11730772.53666386],
    [-12510.48091662534,-2740.752570020901,-285492.8478112217,-98495.79548512613,-2632527.346250018,
     -1205034.497888340,-15137019.29939817,-8560765.912081749,-64550174.40362200,-43305436.93806978],
    [-27103.58303477956,-5719.831450478403,-735179.4395018670,-241305.3893170576,-7842282.258628712,
     -3389586.640563044,-51179318.48413728,-27187309.51284942,-244144745.5117011,-153247217.2931317],
])
Q_PREFACTOR = np.array([
    2.221441469079183, 0.5553603672697958, 0.2082601377261734, 0.08677505738590559,
    0.03796408760633369, 0.01708383942285016, 0.007830093068806324, 0.003635400353374365,
    0.001704093915644234, 0.0008047110157208882, 0.0003822377324674218, 0.0001824316450412695,
    0.0000874151632489416
])
X_THRESHOLD1 = np.array([0.94,0.956,0.968,0.98,0.98,0.983,0.987,0.989,0.99,0.992,0.992,0.9928,0.993])
X_THRESHOLD0 = np.array([0.72,0.72,0.80,0.80,0.83,0.86,0.85,0.88,0.88,0.88,0.885,0.90,0.91])

def _hypergeom_m(m, x):
    if x < X_THRESHOLD1[m]:
        A = HYPERGEOM_0[m] if x < X_THRESHOLD0[m] else HYPERGEOM_I[m]
        xA8 = x + A[8]
        xA6 = x + A[6] + A[7] / xA8
        xA4 = x + A[4] + A[5] / xA6
        xA2 = x + A[2] + A[3] / xA4
        F = A[0] + A[1] / xA2
        return F
    else:
        A = HYPERGEOM_1[m]
        y = 1.0 - x
        y2 = y*y
        z = jnp.log(jnp.maximum(y, EPSILON))
        F = (A[0] + A[1]*z +
             (A[2] + A[3]*z) * y +
             (A[4] + A[5]*z + (A[6] + A[7]*z) * y + (A[8] + A[9]*z) * y2) * y2)
        return F

def legendreQ(n: float, x: ArrayLike) -> np.ndarray:
    xa = np.asarray(x, float)
    xa = np.maximum(xa, 1.0)
    out = np.empty_like(xa)
    m = int(round(n + 0.5))
    is_halfint = (abs(m - (n + 0.5)) < 1e-12) and (0 <= m <= MMAX_HYPERGEOM)
    it = np.nditer(xa, flags=['multi_index'])
    while not it.finished:
        print(it.multi_index)
        xv = float(it[0])
        if is_halfint:
            pref = Q_PREFACTOR[m] / np.sqrt(xv) / (xv**m)
            F = _hypergeom_m(m, 1.0/(xv*xv))
            out[it.multi_index] = pref * F
        else:
            C = (2.0 * xv) ** (-1.0 - n) * np.sqrt(np.pi) * gamma(n + 1.0) / gamma(n + 1.5)
            out[it.multi_index] = C * hyp2f1(1.0 + n/2.0, 0.5 + n/2.0, 1.5 + n, 1.0/(xv*xv))
        it.iternext()
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
        self._rho_m_grid: Dict[int, Array] = {}
        self._rho_m_interp: Dict[int, RegularGridInterpolator] = {}
        self._Phi_m_grid: Dict[int, Array] = {}
        self._Phi_m_interp: Dict[int, RegularGridInterpolator] = {}

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
            self._rho_m_interp[m] = RegularGridInterpolator(
                (self.R, self.Z_nonneg), rho_m[m], method="cubic",
                bounds_error=False, fill_value=None
            )


    def rho_m_eval(self, m: int, R: ArrayLike, z: ArrayLike) -> Array:
        if m < 0 or m > self.mmax: raise ValueError(f"m∈[0,{self.mmax}]")
        if m not in self._rho_m_interp: raise RuntimeError("compute_rho_m() first.")
        Rb = np.asarray(R, float)
        zb = np.abs(np.asarray(z, float))
        shape = Rb.shape
        # pts = np.stack(np.broadcast_arrays(Rb, zb), axis=-1).reshape(-1, 2)
        pts = np.column_stack((Rb.ravel(), zb.ravel()))
        # print('hallo')
        # print(np.broadcast_arrays(Rb, zb))
        return self._rho_m_interp[m](pts).reshape(shape)

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
            self._Phi_m_interp[m]=RegularGridInterpolator(
                (self.R, self.Z_nonneg), Phi, method="cubic", bounds_error=False, fill_value=None
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
        if not self._rho_m_interp:
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

            for i, R0 in tqdm(enumerate(self.R), leave=True):
                for j, z0 in enumerate(self.Z_nonneg):
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
            self._Phi_m_interp[m] = RegularGridInterpolator(
                (self.R, self.Z_nonneg), Phi, method="cubic",
                bounds_error=False, fill_value=None
            )

        return {
            "n_xi": n_xi,
            "n_eta": n_eta,
            "total_nodes": n_xi * n_eta,
            "rule": "simpson([0,1]^2) with log-mapping; Xi(+z')+Xi(−z')",
        }


    # --------- evaluate Φ from Φ_m interpolators ---------
    def phi_m_eval(self, m: int, R: ArrayLike, z: ArrayLike) -> Array:
        if m not in self._Phi_m_interp:
            raise RuntimeError("compute_phi_m_grid_*() and build of Φ_m interpolators required first.")
        Rb = np.asarray(R, float); zb = np.abs(np.asarray(z, float))
        # pts = np.stack(np.broadcast_arrays(Rb, zb), axis=-1).reshape(-1, 2)
        shape = Rb.shape
        pts = np.column_stack((Rb.ravel(), zb.ravel()))
        return self._Phi_m_interp[m](pts).reshape(shape)

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
        if not self._rho_m_interp:
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

