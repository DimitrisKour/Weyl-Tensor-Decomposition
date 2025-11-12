import sympy as sp
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from plotting import *

# Coordinates and parameters
t, r, th, ph, M, a = sp.symbols('t r theta phi M a', real=True)
Sigma  = r**2 + a**2*sp.cos(th)**2
Delta  = r**2 - 2*M*r + a**2

# Metric
g_tt     = -(1 - 2*M*r/Sigma)
g_tphi   = -2*M*a*r*sp.sin(th)**2 / Sigma
g_rr     = Sigma / Delta
g_thth   = Sigma
g_phiphi = (r**2 + a**2 + 2*M*a**2*r*sp.sin(th)**2/Sigma) * sp.sin(th)**2

g_cov = sp.Matrix([
    [g_tt, 0, 0, g_tphi],
    [0, g_rr, 0, 0],
    [0, 0, g_thth, 0],
    [g_tphi, 0, 0, g_phiphi]
])
g_contra = sp.simplify(g_cov.inv())

# Derivatives and lambdify (all these must be inside the file!)
dg = [[[sp.diff(g_cov[i,j], q) for q in (t,r,th,ph)] for j in range(4)] for i in range(4)]
d2g = [[[[sp.diff(dg[i][j][k], q) for q in (t,r,th,ph)] for k in range(4)] for j in range(4)] for i in range(4)]
vars_l = (t, r, th, ph, M, a)
g_fun    = sp.lambdify(vars_l, g_cov, "numpy")
gInv_fun = sp.lambdify(vars_l, g_contra, "numpy")
dg_fun   = [[[sp.lambdify(vars_l, dg[i][j][k], "numpy") for k in range(4)] for j in range(4)] for i in range(4)]
d2g_fun  = [[[[sp.lambdify(vars_l, d2g[i][j][k][l], "numpy") for l in range(4)] for k in range(4)] for j in range(4)] for i in range(4)]

# ...then all your functions (eval_metric_and_derivs, christoffel, etc.)


def eval_metric_and_derivs(t_, r_, th_, ph_, M_, a_):
    g  = np.array(g_fun(t_, r_, th_, ph_, M_, a_), dtype=float)
    gi = np.array(gInv_fun(t_, r_, th_, ph_, M_, a_), dtype=float)
    dg_arr  = np.zeros((4,4,4), dtype=float)
    d2g_arr = np.zeros((4,4,4,4), dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                dg_arr[i,j,k] = float(dg_fun[i][j][k](t_, r_, th_, ph_, M_, a_))
                for l in range(4):
                    d2g_arr[i,j,k,l] = float(d2g_fun[i][j][k][l](t_, r_, th_, ph_, M_, a_))
    return g, gi, dg_arr, d2g_arr

# -----------------------------
# 2) Γ, ∂Γ, Riemann, Ricci, R, Weyl
# -----------------------------
def christoffel(g, gi, dg):
    # Γ^ρ_{μν} = 1/2 g^{ρσ} (∂_μ g_{σν} + ∂_ν g_{σμ} - ∂_σ g_{μν})
    Gam = np.zeros((4,4,4), dtype=float)
    for rho in range(4):
        for mu in range(4):
            for nu in range(4):
                s = 0.0
                for sig in range(4):
                    s += gi[rho, sig]*(dg[sig, nu, mu] + dg[sig, mu, nu] - dg[mu, nu, sig])
                Gam[rho, mu, nu] = 0.5*s
    return Gam

def d_christoffel(Gam, gi, g, dg, d2g):
    # ∂_λ Γ^ρ_{μν} computed by differentiating definition (use d2g + dg*gi*dg terms)
    # We differentiate the explicit expression above: treat gi also λ-dependent.
    # ∂_λ g^{ρσ} = - g^{ρα} (∂_λ g_{αβ}) g^{βσ}
    dGam = np.zeros((4,4,4,4), dtype=float)  # [rho, mu, nu, lam]
    dgi = np.zeros((4,4,4), dtype=float)
    for lam in range(4):
        for r1 in range(4):
            for s1 in range(4):
                tmp = 0.0
                for a1 in range(4):
                    for b1 in range(4):
                        tmp += -gi[r1,a1]*dg[a1,b1,lam]*gi[b1,s1]
                dgi[r1,s1,lam] = tmp

    for lam in range(4):
        for rho in range(4):
            for mu in range(4):
                for nu in range(4):
                    s = 0.0
                    for sig in range(4):
                        # derivative acts on gi and on the bracketed (dg + dg - dg) part
                        bracket = (dg[sig,nu,mu] + dg[sig,mu,nu] - dg[mu,nu,sig])
                        dbracket = (d2g[sig,nu,mu,lam] + d2g[sig,mu,nu,lam] - d2g[mu,nu,sig,lam])
                        s += 0.5*( dgi[rho,sig,lam]*bracket + gi[rho,sig]*dbracket )
                    dGam[rho,mu,nu,lam] = s
    return dGam

def riemann(g, gi, Gam, dGam):
    # R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} - Γ^ρ_{νλ}Γ^λ_{μσ}
    R = np.zeros((4,4,4,4), dtype=float)
    for rho in range(4):
        for sig in range(4):
            for mu in range(4):
                for nu in range(4):
                    term = dGam[rho,nu,sig,mu] - dGam[rho,mu,sig,nu]
                    for lam in range(4):
                        term += Gam[rho,mu,lam]*Gam[lam,nu,sig] - Gam[rho,nu,lam]*Gam[lam,mu,sig]
                    R[rho,sig,mu,nu] = term
    return R

def lower_all_R(R, g):
    # R_{αβγδ} = g_{αρ} R^ρ_{βγδ}
    Rlow = np.zeros((4,4,4,4), dtype=float)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    s = 0.0
                    for r1 in range(4):
                        s += g[a,r1]*R[r1,b,c,d]
                    Rlow[a,b,c,d] = s
    return Rlow

def ricci_and_scalar(Rlow, gi):
    # Ricci: R_{bd} = R^a_{bad} = g^{aα} R_{α b a d}
    Ric = np.zeros((4,4), dtype=float)
    for b in range(4):
        for d in range(4):
            s = 0.0
            for a in range(4):
                for alp in range(4):
                    s += gi[a,alp]*Rlow[alp,b,a,d]
            Ric[b,d] = s
    # Scalar R = g^{bd} R_{bd}
    scal = 0.0
    for b in range(4):
        for d in range(4):
            scal += gi[b,d]*Ric[b,d]
    return Ric, scal

def weyl_from_R(Rlow, g, gi):
    # In 4D: Schouten S_ab = 1/2 (R_ab - (R/6) g_ab)
    Ric, Rsc = ricci_and_scalar(Rlow, gi)
    S = 0.5*(Ric - (Rsc/6.0)*g)
    # C_{abcd} = R_{abcd} - 2( g_{a[c} S_{d]b} - g_{b[c} S_{d]a} )
    # Implement antisymmetrization in [cd]
    C = np.zeros((4,4,4,4), dtype=float)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    term = Rlow[a,b,c,d]
                    term -= 2.0*( g[a,c]*S[d,b] - g[a,d]*S[c,b] - g[b,c]*S[d,a] + g[b,d]*S[c,a] )
                    C[a,b,c,d] = term
    return C

# -----------------------------
# 3) Levi-Civita (tensor) & E,B
# -----------------------------
def levi_civita_down(g):
    # ε_{abcd} with (-,+,+,+) signature convention; |det g|^1/2 factor
    # Take orientation (t,r,th,ph) -> +1
    eps0123 = 1.0
    sgn = np.sign(np.linalg.det(g))
    vol = np.sqrt(abs(np.linalg.det(g)))
    # Build from symbol ε_{abcd} in coordinate basis:
    # We'll generate by antisym permutations with ε_{0123}=+1
    eps = np.zeros((4,4,4,4), dtype=float)
    import itertools
    for p in itertools.permutations([0,1,2,3]):
        # parity of permutation:
        parity = ( (p[0]>p[1]) + (p[0]>p[2]) + (p[0]>p[3]) +
                   (p[1]>p[2]) + (p[1]>p[3]) + (p[2]>p[3]) ) % 2
        eps[p] = ( -1.0 if parity else 1.0 ) * eps0123 * vol
    # sgn is typically negative with (-,+,+,+); vol is positive; eps already accounts for vol.
    return eps

def raise_indices(tensor, gi, n_up):
    """
    Raise 'n_up' indices of a (0,k) tensor using the metric inverse gi.
    Currently supports rank-2 tensors.
    """
    T = tensor.copy()
    if n_up == 1:
        # Raise first index: T^a{}_b = g^{ac} T_cb
        return np.einsum("ac,cb->ab", gi, T)
    elif n_up == 2:
        # Raise both: T^{ab} = g^{ac} g^{bd} T_cd
        return np.einsum("ac,bd,cd->ab", gi, gi, T)
    else:
        raise ValueError("Only supports n_up = 1 or 2 for rank-2 tensors")

def EB_from_Weyl(C, g, gi, u):
    # E_ab = C_{acbd} u^c u^d ;  B_ab = 1/2 ε_{ac}{}^{ef} C_{efbd} u^c u^d
    # Build ε_{ac}{}^{ef} = ε_{acmn} g^{me} g^{nf}
    eps = levi_civita_down(g)
    # build ε_{ac}{}^{ef}
    eps_mixed = np.zeros((4,4,4,4), dtype=float)
    for a in range(4):
        for c in range(4):
            for e in range(4):
                for f in range(4):
                    s = 0.0
                    for m in range(4):
                        for n in range(4):
                            s += eps[a,c,m,n]*gi[m,e]*gi[n,f]
                    eps_mixed[a,c,e,f] = s

    # Contract u^c u^d:
    E = np.zeros((4,4), dtype=float)
    B = np.zeros((4,4), dtype=float)
    for a in range(4):
        for b in range(4):
            sE = 0.0
            sB = 0.0
            for c in range(4):
                for d in range(4):
                    sE += C[a,c,b,d]*u[c]*u[d]
                    # B_ab
                    s_tmp = 0.0
                    for e in range(4):
                        for f in range(4):
                            s_tmp += eps_mixed[a,c,e,f]*C[e,f,b,d]
                    sB += 0.5*s_tmp*u[c]*u[d]
            E[a,b] = sE
            B[a,b] = sB

    # Raise to get E^{ab}, B^{ab}
    E_up = E.copy()
    B_up = B.copy()
    # raise both indices:
    E_up = raise_indices(E, gi, 2)
    B_up = raise_indices(B, gi, 2)
    return E, B, E_up, B_up

def invariants_from_EB(E, B, E_up, B_up):
    # Scalars on the u^a-rest-space foliation:
    E2 = 0.0; B2 = 0.0; EB = 0.0
    for a in range(4):
        for b in range(4):
            E2 += E[a,b]*E_up[a,b]
            B2 += B[a,b]*B_up[a,b]
            EB += E[a,b]*B_up[a,b]
    # Your J1, J2 (up to convention factors):
    J1 = E2 - B2
    J2 = EB
    return E2, B2, J1, J2

# -----------------------------
# 4) Observers
# -----------------------------
def u_static(g):
    # Only valid where g_tt<0 (outside ergosphere). u^t = 1/sqrt(-g_tt)
    ut = 1.0/np.sqrt(-g[0,0])
    u = np.array([ut, 0.0, 0.0, 0.0])
    return u

def u_with_Omega(g, gi, Omega):
    # Circular (t,phi) observer: u^μ ∝ (1,0,0,Ω), normalize:
    A = g[0,0] + 2*Omega*g[0,3] + Omega**2 * g[3,3]
    ut = 1.0/np.sqrt(-A)
    return np.array([ut, 0.0, 0.0, Omega*ut])

def Omega_ZAMO(g):
    # ZAMO angular velocity: Ω = -g_{tφ}/g_{φφ}
    return -g[0,3]/g[3,3]

def Omega_geodesic_equatorial(r_, M_, a_, prograde=True):
    # Keplerian Ω_± at θ=π/2: Ω = ± M^{1/2} / (r^{3/2} ± a M^{1/2})
    s = +1.0 if prograde else -1.0
    return s*np.sqrt(M_) / (r_**1.5 + s*a_*np.sqrt(M_))

# -----------------------------
# 5) Convenience evaluator
# -----------------------------
def EB_at_point(r_, th_, M_, a_, obs='static', prograde=True, t_=0.0, ph_=0.0):
    g, gi, dg, d2g = eval_metric_and_derivs(t_, r_, th_, ph_, M_, a_)
    Gam  = christoffel(g, gi, dg)
    dGam = d_christoffel(Gam, gi, g, dg, d2g)
    R    = riemann(g, gi, Gam, dGam)
    Rlow = lower_all_R(R, g)
    C    = weyl_from_R(Rlow, g, gi)

    if obs=='static':
        u = u_static(g)
    elif obs=='zamo':
        Om = Omega_ZAMO(g)
        u  = u_with_Omega(g, gi, Om)
    elif obs in ('prograde','retrograde'):
        Om = Omega_geodesic_equatorial(r_, M_, a_, prograde=(obs=='prograde'))
        u  = u_with_Omega(g, gi, Om)
    else:
        raise ValueError("obs must be 'static', 'zamo', 'prograde', 'retrograde'.")

    E, B, E_up, B_up = EB_from_Weyl(C, g, gi, u)
    E2, B2, J1, J2   = invariants_from_EB(E, B, E_up, B_up)
    return dict(E=E, B=B, E2=E2, B2=B2, J1=J1, J2=J2, u=u, g=g)

# ----------------- Ensure timelike observers -----------------
def r_plus(M, a):
    return M + np.sqrt(max(M*M - a*a, 0.0))

def r_erg(theta, M, a):
    return M + np.sqrt(max(M*M - (a*np.cos(theta))**2, 0.0))

def r_photon_eq(M, a, prograde=True):
    s = +1 if prograde else -1
    chi = a / M
    # r_ph^± (equatorial)
    return 2*M*(1 + np.cos((2/3)*np.arccos(-s*chi)))

def r_mb_eq(M, a, prograde=True):
    s = +1 if prograde else -1
    return 2*M - s*a + 2*np.sqrt(M*(M - s*a))

def r_isco_eq(M, a, prograde=True):
    chi = a / M
    Z1 = 1 + (1 - chi**2)**(1/3) * ((1 + chi)**(1/3) + (1 - chi)**(1/3))
    Z2 = np.sqrt(3*chi**2 + Z1**2)
    if prograde:
        return M*(3 + Z2 - np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))
    else:
        return M*(3 + Z2 + np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))

def recommend_r_grid(observer, M=1.0, a=0.9, theta=np.pi/2, r_max=30.0, N=400, eps=1e-3):
    """
    Returns a safe, monotonic r-array for the chosen observer.
    observer ∈ {'static','zamo','prograde','retrograde'}
    """
    if observer == 'static':
        r_min = max(r_erg(theta, M, a) + eps, r_plus(M, a) + eps)  # ergosurface dominates
    elif observer == 'zamo':
        r_min = r_plus(M, a) + eps
    elif observer == 'prograde':
        r_min = r_isco_eq(M, a, prograde=True) + eps
    elif observer == 'retrograde':
        r_min = r_isco_eq(M, a, prograde=False) + eps
    else:
        raise ValueError("observer must be 'static', 'zamo', 'prograde', or 'retrograde'.")

    # Guard: if r_min ≥ r_max, expand r_max
    if r_min >= r_max:
        r_max = r_min + 10.0

    # Use geometric spacing near r_min for better detail, then linear tail
    r1 = np.geomspace(r_min, (r_min + r_max)/2, N//2)
    r2 = np.linspace((r_min + r_max)/2, r_max, N - N//2)
    r_vals = np.unique(np.concatenate([r1, r2]))

    # Optional: filter any radii where chosen u^a would not be timelike
    # (mostly redundant if bounds above are used)
    safe = []
    for rv in r_vals:
        out_g = EB_at_point(rv, theta, M, a, obs=observer)  # builds u and normalizes
        # If normalization succeeded, u·u = -1; if not, EB_at_point would raise
        safe.append(rv)
    return np.array(safe)


# ------------------------ Plotting ------------------------

def auto_linthresh(arr):
    arr = np.asarray(arr)
    # smallest non-zero magnitude
    nz = np.nanmin(np.abs(arr[np.nonzero(arr)]))
    return nz

def lighten_ticks_keep_zero(ax, alpha=0.35, fontsize=9):
    # Ensure ticks exist
    ax.figure.canvas.draw()

    # Get numeric tick values (these are reliable)
    ticks = ax.get_yticks()
    labels = ax.yaxis.get_ticklabels()

    for val, label in zip(ticks, labels):
        if np.isclose(val, 0.0, atol=1e-14):   # zero tick
            label.set_alpha(1.0)
            label.set_fontsize(fontsize)
        else:
            label.set_alpha(alpha)
            label.set_fontsize(fontsize)

def apply_sci_ticks(ax, digits=2):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))   # keep standard form except for very big/small
    fmt.set_scientific(True)
    ax.yaxis.set_major_formatter(fmt)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(9)  # mild reduction

def plot_Js(r_vals, J1, J2, title=None):
    fig, ax = plt.subplots(1, 2, figsize=(11.1, 5.3), tight_layout=True)

    lt1 = auto_linthresh(J1)
    lt2 = auto_linthresh(J2)

    # J₁
    ax[0].plot(r_vals, J1, color='black')
    ax[0].axhline(0, color='black', lw=0.8)
    ax[0].set_xlabel(r'$r/M$'); ax[0].set_ylabel(r'$J_1$')
    ax[0].set_title(r'$J_1(r)$')
    ax[0].set_yscale("symlog", linthresh=lt1)
    lighten_ticks_keep_zero(ax[0])
    ax[0].grid(True, alpha=0.3)

    # J₂
    ax[1].plot(r_vals, J2, color='black')
    ax[1].axhline(0, color='black', lw=0.8)
    ax[1].set_xlabel(r'$r/M$'); ax[1].set_ylabel(r'$J_2$')
    ax[1].set_title(r'$J_2(r)$')
    ax[1].set_yscale("symlog", linthresh=lt2)
    lighten_ticks_keep_zero(ax[1])
    ax[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, y=0.89, fontsize=20)

def plot_ratios(r_vals, E2, B2, J1, J2, title=None, eps=1e-14, legend_loc1="center right", legend_loc2="center right"):
    """
    Diagnostic ratio plots with neutral aesthetics.
    Left: E²/|J₁|, B²/|J₁| and overlay J₁.
    Right: E²/B² and overlay J₂/J₁.

    legend_loc1 : str
        Legend location for first axis (e.g. "center right", "upper left", etc.).
    legend_loc2 : str
        Legend location for second axis (e.g. "center right", "upper left", etc.).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Safe divisions ---
    J1_safe = np.where(np.abs(J1) < eps, np.nan, J1)
    E2_by_J1 = E2 / np.abs(J1_safe)
    B2_by_J1 = B2 / np.abs(J1_safe)
    EB_ratio = E2 / B2
    J2_by_J1 = J2 / J1_safe

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), tight_layout=True)

    # ================================================================
    # LEFT PANEL
    # ================================================================
    l1 = ax[0].plot(r_vals, E2_by_J1, color='#1b4d5a', label=r'$E^2/|J_1|$')[0]
    l2 = ax[0].plot(r_vals, B2_by_J1, color='#8c735a', ls='--', label=r'$B^2/|J_1|$')[0]
    ax0b = ax[0].twinx()
    l3 = ax0b.plot(r_vals, J1, color='#444444', lw=1.0, ls=':', label=r'$J_1$')[0]

    ax[0].set_yscale("symlog", linthresh=1e-6)
    ax[0].set_xlabel(r'$r/M$')
    ax[0].set_ylabel(r'Normalized values')
    ax0b.set_ylabel(r'$J_1$')
    ax[0].grid(True, alpha=0.3)

    leg0 = ax[0].legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()],
                        frameon=False, handlelength=1.7, loc=legend_loc1)
    leg0.set_zorder(1000)

    ymin = np.nanmin([E2_by_J1.min(), B2_by_J1.min()]) * 0.5
    ymax = np.nanmax([E2_by_J1.max(), B2_by_J1.max()]) * 2
    ax[0].fill_between(r_vals, ymin, ymax, where=(J1 > 0), color='#d3d3d3', alpha=0.25)
    ax[0].fill_between(r_vals, ymin, ymax, where=(J1 < 0), color='#a3c1da', alpha=0.25)

    # ================================================================
    # RIGHT PANEL
    # ================================================================
    l4 = ax[1].plot(r_vals, EB_ratio, color='black', label=r'$E^2/B^2$')[0]
    ax2 = ax[1].twinx()
    l5 = ax2.plot(r_vals, J2_by_J1, color='#6e6e6e', ls=':', label=r'$J_2/J_1$')[0]

    ax[1].set_xlabel(r'$r/M$')
    ax[1].set_ylabel(r'$E^2/B^2$')
    ax2.set_ylabel(r'$J_2/J_1$')
    ax[1].set_yscale("log")
    ax[1].grid(True, alpha=0.3)

    leg1 = ax2.legend([l4, l5], [l4.get_label(), l5.get_label()],
                      frameon=False, handlelength=1.7, loc=legend_loc2)
    leg1.set_zorder(1000)

    # ================================================================
    if title:
        fig.suptitle(title, y=0.89, fontsize=20)

    return fig, ax

def compare_observers(Mval=1.0, aval=0.9, theta=np.pi/2,
                      global_rmax_cap=40.0, N=400, eps=1e-3, rmax_factor=10.0):
    """
    Computes curvature quantities for 'static', 'prograde', 'retrograde' observers,
    automatically selecting physically valid r-ranges via recommend_r_grid(),
    and setting r_max = min(global_rmax_cap, rmax_factor * r_min).

    Parameters
    ----------
    Mval : float
        Black hole mass parameter.
    aval : float
        Spin parameter (|a| <= M).
    theta : float
        Polar angle (rad).
    global_rmax_cap : float
        Absolute cap on outer radius (in M units).
    N : int
        Number of sample points.
    eps : float
        Small safety offset above limiting radius.
    rmax_factor : float
        Multiple of r_min used to define local r_max.

    Returns
    -------
    results : dict
        results[observer] = {'r', 'E2', 'B2', 'J1', 'J2', 'r_min', 'r_max'}
    """

    observers = ['static', 'prograde', 'retrograde']
    results = {}

    for obs in observers:
        # --- get preliminary safe lower bound ---
        r_min_guess = recommend_r_grid(observer=obs, M=Mval, a=aval,
                                       theta=theta, r_max=10.0, N=5, eps=eps)[0]

        # --- define r_max relative to r_min ---
        r_max_local = min(global_rmax_cap, rmax_factor * r_min_guess)

        # --- recompute full grid now that r_max_local is set ---
        rs = recommend_r_grid(observer=obs, M=Mval, a=aval,
                              theta=theta, r_max=r_max_local, N=N, eps=eps)

        # --- compute along this grid ---
        E2, B2, J1, J2 = [], [], [], []
        for rv in rs:
            out = EB_at_point(rv, theta, Mval, aval, obs=obs)
            E2.append(out['E2']); B2.append(out['B2'])
            J1.append(out['J1']); J2.append(out['J2'])

        results[obs] = dict(
            r=rs,
            E2=np.array(E2),
            B2=np.array(B2),
            J1=np.array(J1),
            J2=np.array(J2),
            r_min=float(rs[0]),
            r_max=float(rs[-1])
        )

    return results