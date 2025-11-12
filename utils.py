# utils.py
from __future__ import annotations
import sympy as sp
from sympy import sin, cos, sqrt, I
from sympy import Matrix
import itertools
from einsteinpy.symbolic import (
    MetricTensor,
    WeylTensor,
    RiemannCurvatureTensor,
    RicciTensor,
    RicciScalar,
)
from IPython.display import display, Math

# ---------- Core helpers ----------
def metric_from_components(components, coords):
    """
    components: 2D sympy Matrix or nested list (covariant g_{ab})
    coords: tuple/list of sympy symbols (x0, x1, x2, x3)
    returns: EinsteinPy MetricTensor (covariant)
    """
    g = Matrix(components)
    return MetricTensor(g.tolist(), tuple(coords))

def inv_metric(metric: MetricTensor):
    """Return g^{ab} as sympy Matrix."""
    return Matrix(metric.tensor()).inv()

def raise_index(t_ab, g_inv):
    """Raise one index: t^a_b = g^{ac} t_{cb}. Input/return as sympy Matrix."""
    return g_inv * Matrix(t_ab)

def lower_index(t_ab, g):
    """Lower one index: t_{ab} = g_{ac} t^c_b."""
    return Matrix(g) * Matrix(t_ab)

def eps_symbol(a, b, c, d):
    """
    Levi-Civita symbol ε_abcd with ε_0123 = +1.
    Returns 0 if any indices repeat.
    """
    inds = [a, b, c, d]
    # zero if any repeated
    if len(set(inds)) < 4:
        return sp.Integer(0)
    # parity of permutation (number of inversions)
    inversions = sum(
        1 for i, j in itertools.combinations(range(4), 2)
        if inds[i] > inds[j]
    )
    sign = -1 if inversions % 2 else 1
    return sp.Integer(sign)

def levi_civita_eps(g):
    """
    Levi-Civita tensor density sqrt(|g|)*ε_abcd.
    Returns (eps_symbol, sqrt_det_g).
    """
    gM = Matrix(g)
    detg = sp.simplify(gM.det())
    detg_sqrt = sp.sqrt(abs(detg))
    return eps_symbol, detg_sqrt

def hodge_dual_weyl(C_abcd, g, g_inv):
    """
    Left Hodge dual on the first index pair: (*C)_{ab cd} = 1/2 epsilon_{ab}{}^{mn} C_{mn cd}
    Returns a 4-index dict (*C) with keys (a,b,c,d).
    """
    n = g.shape[0]
    eps_symbol, detg_sqrt = levi_civita_eps(g)
    # epsilon_{ab}{}^{mn} = detg_sqrt * eps_symbol(a,b,m,n) * g^{mp} g^{nq} (to raise indices)
    Cstar = {}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    s = 0
                    for m in range(n):
                        for n2 in range(n):
                            # raise m,n on eps: eps_{abmn} g^{mp} g^{nq}
                            for p in range(n):
                                for q in range(n):
                                    eps_abmn = detg_sqrt * eps_symbol(a,b,m,n2)
                                    eps_ab_pq = eps_abmn * g_inv[m, p] * g_inv[n2, q]
                                    s += sp.Rational(1,2) * eps_ab_pq * C_abcd[(p,q,c,d)]
                    Cstar[(a,b,c,d)] = sp.simplify(s)
    return Cstar

def print_tensor_components(T, name="T", tol=0, return_df=False):
    """
    Print and/or return unique nonzero components of a 4-index tensor dict {(a,b,c,d): value}.
    
    Parameters
    ----------
    T : dict
        Tensor components as {(a,b,c,d): sympy expression}.
    name : str
        Label used when printing.
    tol : float, optional
        Skip components whose absolute value <= tol (after simplification).
    return_df : bool, optional
        If True, returns a pandas DataFrame of the nonzero components.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with columns [a,b,c,d,value] if return_df=True.
    """
    import pandas as pd
    shown = set()
    rows = []
    for (a, b, c, d), val in sorted(T.items()):
        # Weyl symmetries: antisymmetric in (a,b) and (c,d), symmetric under pair exchange
        if a > b or c > d:
            continue
        if (c, d, a, b) in shown:
            continue
        val_simpl = sp.simplify(val)
        if val_simpl == 0:
            continue
        if tol and abs(val_simpl) <= tol:
            continue
        shown.add((a, b, c, d))
        latex_val = sp.latex(val_simpl)
        display(Math(rf"{name}_{{{a}{b}{c}{d}}} = {latex_val}"))
        rows.append((a, b, c, d, val_simpl))
    
    if not rows:
        print(f"All {name} components are zero or redundant.")
        if return_df:
            return pd.DataFrame(columns=["a","b","c","d","value"])
        return None
    
    if return_df:
        df = pd.DataFrame(rows, columns=["a","b","c","d","value"])
        return df

# ---------- Curvature ----------
def weyl_from_metric(metric):
    """
    Symbolic Weyl tensor C_{abcd} (all covariant) via EinsteinPy.
    Returns dict {(a,b,c,d): sympy expr}.
    """
    import sympy as sp
    from einsteinpy.symbolic import WeylTensor, MetricTensor

    # Ensure metric is EinsteinPy MetricTensor
    if not isinstance(metric, MetricTensor):
        raise TypeError("Input must be an EinsteinPy MetricTensor object")

    # Build Weyl tensor and ensure all indices are lowered
    C = WeylTensor.from_metric(metric).change_config("llll")

    # Extract components
    arr = sp.Array(C.tensor())   # C_{abcd}
    shape = arr.shape
    out = {}

    for a in range(shape[0]):
        for b in range(shape[1]):
            for c in range(shape[2]):
                for d in range(shape[3]):
                    out[(a, b, c, d)] = sp.simplify(arr[a, b, c, d])

    return out

def riemann_from_metric(metric: MetricTensor):
    R = RiemannCurvatureTensor.from_metric(metric).tensor()
    n = len(R)
    out = {}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    out[(a,b,c,d)] = sp.simplify(R[a][b][c][d])
    return out

def ricci_from_metric(metric: MetricTensor):
    Ric = RicciTensor.from_metric(metric).tensor()
    n = len(Ric)
    return Matrix([[sp.simplify(Ric[i][j]) for j in range(n)] for i in range(n)])

def ricci_scalar(metric: MetricTensor):
    return sp.simplify(RicciScalar.from_metric(metric).expr)

# ---------- Projections: E_ab, B_ab ----------
def electric_magnetic_weyl(C, metric: MetricTensor, u_cov):
    """
    E_ab = C_{acbd} u^c u^d
    B_ab = (*C)_{acbd} u^c u^d, with * the left Hodge dual on the first pair.
    Inputs:
      - C: dict {(a,b,c,d): C_abcd} (covariant)
      - metric: EinsteinPy MetricTensor (covariant)
      - u_cov: covariant 4-velocity components as a sympy Matrix/list [u_0,...,u_3]
    Returns: (E_ab Matrix, B_ab Matrix)
    """
    g = Matrix(metric.tensor())
    g_inv = g.inv()
    n = g.shape[0]
    u_cov = Matrix(u_cov)
    # raise to get u^a
    u_up = g_inv * u_cov

    # Hodge dual *C on first index pair
    Cstar = hodge_dual_weyl(C, g, g_inv)

    def project(Clike):
        M = sp.MutableDenseMatrix(n, n, [0]*n*n)
        for a in range(n):
            for b in range(n):
                s = 0
                for c in range(n):
                    for d in range(n):
                        s += Clike[(a,c,b,d)] * u_up[c] * u_up[d]
                M[a,b] = sp.simplify(s)
        # Make symmetric-traceless explicitly (numerical stability in later use)
        M = sp.simplify((M + M.T) / 2)
        tr = sp.simplify((g_inv * M).trace())
        M = sp.simplify(M - tr * g / n)  # in 4D, E and B are tracefree; this enforces it algebraically
        return M

    E = project(C)
    B = project(Cstar)

    # pretty display
    display(Math(r"E_{ab} = " + sp.latex(E)))
    display(Math(r"B_{ab} = " + sp.latex(B)))
    
    return E, B

def eigenstructure_EB(E, B, show_vectors=False, compute_Q=False):
    """
    Compute and display eigenvalues/eigenvectors for E_ab, B_ab,
    and optionally for Q_ab = E_ab + i B_ab.

    Parameters
    ----------
    E, B : sympy.Matrix
        Electric and magnetic parts of the Weyl tensor.
    show_vectors : bool, optional
        If True, also display eigenvectors along with eigenvalues.
    compute_Q : bool, optional
        If True (default), compute eigenstructure of Q = E + i B.

    Returns
    -------
    dict with:
        'Evals', 'Evecs', 'Bevals', 'Bevecs'
        optionally 'Qevals', 'Qevecs' if compute_Q=True

    Notes
    -----
    - Q_ab encodes the complex Weyl tensor and determines the Petrov type.
    - For static spacetimes (like Schwarzschild), B_ab = 0 so Q = E.
    """
    # E tensor
    E_evals = E.eigenvals()
    E_evecs = E.eigenvects()

    # B tensor
    B_evals = B.eigenvals()
    B_evecs = B.eigenvects()

    # Display eigenvalues (LaTeX)
    display(Math(r"E\text{-tensor eigenvalues: }" + sp.latex(E_evals)))
    display(Math(r"B\text{-tensor eigenvalues: }" + sp.latex(B_evals)))

    # Optionally compute Q
    if compute_Q:
        Q = E + sp.I * B
        Q_evals = Q.eigenvals()
        Q_evecs = Q.eigenvects()
        display(Math(r"Q = E + iB\text{ eigenvalues: }" + sp.latex(Q_evals)))
    else:
        Q_evals, Q_evecs = None, None

    # Optional eigenvector display
    if show_vectors:
        def show_vecs(label, evects):
            lines = []
            for val, mult, vecs in evects:
                for v in vecs:
                    lines.append(rf"\lambda={sp.latex(val)}:\; v={sp.latex(v)}")
            if lines:
                display(Math(label + r":\; " + r",\quad ".join(lines)))

        show_vecs("E", E_evecs)
        show_vecs("B", B_evecs)
        if compute_Q:
            show_vecs("Q", Q_evecs)

    return {
        "Evals": E_evals,
        "Evecs": E_evecs,
        "Bevals": B_evals,
        "Bevecs": B_evecs,
        "Qevals": Q_evals,
        "Qevecs": Q_evecs,
    }


from IPython.display import display, Math

def norm_EB(E, B, metric, show_results=True):
    """
    Compute and (optionally) display the norms of the electric and magnetic
    parts of the Weyl tensor.

        |E|^2 = E_ab E^ab
        |B|^2 = B_ab B^ab
    """
    g = sp.Matrix(metric.tensor())
    g_inv = g.inv()

    # Raise both indices: E^{ab} = g^{ac} g^{bd} E_cd
    E_upup = g_inv * E * g_inv
    B_upup = g_inv * B * g_inv

    # Contract: E_ab E^ab = Σ E_ab * E^ab
    E2 = sp.simplify(sum(E[i, j] * E_upup[i, j] for i in range(E.rows) for j in range(E.cols)))
    B2 = sp.simplify(sum(B[i, j] * B_upup[i, j] for i in range(B.rows) for j in range(B.cols)))

    if show_results:
        display(Math(r"|E|^2 = " + sp.latex(E2) + r", \quad |B|^2 = " + sp.latex(B2)))

    return {"E2": E2, "B2": B2}

def check_trace(E, B, metric):
    """
    Compute and print the traces of E_ab and B_ab with the inverse metric.

    Parameters
    ----------
    E, B : sympy.Matrix
        Electric and magnetic parts of the Weyl tensor (E_ab, B_ab).
    metric : MetricTensor
        EinsteinPy MetricTensor (provides g_ab).

    Returns
    -------
    dict with 'trE', 'trB' : sympy.Expr
    """
    g_cov = Matrix(metric.tensor())
    g_inv = g_cov.inv()
    trE = sp.simplify((g_inv * E).trace())
    trB = sp.simplify((g_inv * B).trace())
    print("trace(E) =", trE)
    print("trace(B) =", trB)
    return {"trE": trE, "trB": trB}


# ---------- Scalar invariants ----------
def weyl_invariants(C, metric: MetricTensor, convention="raw", show_results=True):
    """
    Compute Weyl tensor quadratic invariants with selectable normalization.

    Parameters
    ----------
    C : dict
        Covariant Weyl tensor components {(a,b,c,d): C_abcd}.
    metric : MetricTensor
        EinsteinPy MetricTensor object (covariant metric).
    convention : str, optional
        - 'raw'       : returns (W1, W2) where
                        W1 = C_{abcd} C^{abcd},
                        W2 = C_{abcd} (*C)^{abcd}.
        - 'selfdual'  : returns I = (W1 - i W2)/16 (Penrose–Rindler convention).
        - 'np_package': returns dict {'W1','W2','I'} with the same normalization.

    Returns
    -------
    tuple or sympy expression or dict
        Depending on `convention`.
    """
    g = Matrix(metric.tensor())
    g_inv = g.inv()
    n = g.shape[0]

    # --- Helper: raise all indices on a copy ---
    def raise_all(Cdict):
        C_up = {}
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        s = 0
                        for e in range(n):
                            for f in range(n):
                                for h in range(n):
                                    for k in range(n):
                                        s += (g_inv[a,e]*g_inv[b,f]*g_inv[c,h]*g_inv[d,k]
                                              * Cdict[(e,f,h,k)])
                        C_up[(a,b,c,d)] = sp.simplify(s)
        return C_up

    # Raise indices and compute dual
    C_up = raise_all(C)
    Cstar = hodge_dual_weyl(C, g, g_inv)
    Cstar_up = raise_all(Cstar)

    # --- Compute quadratic invariants ---
    W1 = 0
    W2 = 0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    W1 += sp.simplify(C[(a,b,c,d)] * C_up[(a,b,c,d)])
                    W2 += sp.simplify(C[(a,b,c,d)] * Cstar_up[(a,b,c,d)])
    W1 = sp.simplify(W1)
    W2 = sp.simplify(W2)

    if show_results:
        display(Math(r"J_1 = " + sp.latex(sp.simplify(W1)) + r", \quad J_2 = " + sp.latex(sp.simplify(W2))))

    # --- Conventions ---
    if convention == "raw":
        return W1, W2

    if convention == "selfdual":
        # Penrose–Rindler normalization: I = (W1 - i W2)/16
        I = sp.simplify((W1 - sp.I * W2) / 16)
        return I

    if convention == "np_package":
        I = sp.simplify((W1 - sp.I * W2) / 16)
        return {"W1": W1, "W2": W2, "I": I}

    # fallback
    return W1, W2


# ---------- Bel-Robinson Tensor + Super Energy, Super-Poynting ----------

def bel_robinson_from_weyl(C, metric: MetricTensor):
    """
    Bel–Robinson tensor from the Weyl tensor (vacuum), all indices down:
        T_abcd = C_aecf C_b{}^{e}{}_d{}^{f} + (*C)_aecf (*C)_b{}^{e}{}_d{}^{f}

    Parameters
    ----------
    C : dict
        Covariant Weyl components {(a,b,c,d): C_abcd}.
    metric : MetricTensor

    Returns
    -------
    dict
        Covariant Bel–Robinson components {(a,b,c,d): T_abcd}.
    """
    g = Matrix(metric.tensor())
    g_inv = g.inv()
    n = g.shape[0]

    # left Hodge dual on first index pair
    Cstar = hodge_dual_weyl(C, g, g_inv)

    T = {}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    s = 0
                    # First term: C_{a e c f} * C_{b}{}^{e}{}_{d}{}^{f}
                    for e in range(n):
                        for f in range(n):
                            # raise e,f on the second C via g^{e m} g^{f n}
                            tmp = 0
                            for m in range(n):
                                for n2 in range(n):
                                    tmp += g_inv[e, m] * g_inv[f, n2] * C[(b, m, d, n2)]
                            s += C[(a, e, c, f)] * tmp

                    # Second term: (*C)_{a e c f} * (*C)_{b}{}^{e}{}_{d}{}^{f}
                    for e in range(n):
                        for f in range(n):
                            tmp = 0
                            for m in range(n):
                                for n2 in range(n):
                                    tmp += g_inv[e, m] * g_inv[f, n2] * Cstar[(b, m, d, n2)]
                            s += Cstar[(a, e, c, f)] * tmp

                    T[(a, b, c, d)] = sp.simplify(s)

    return T


def bel_robinson_observer(C, metric: MetricTensor, u_cov, show_results=False):
    """
    Bel–Robinson 'energy' and 'flux' for a timelike observer u^a (u^a u_a = -1, signature -,+,+,+).

    Computes:
        U = T_abcd u^a u^b u^c u^d
        P_a = - h_a{}^e T_ebcd u^b u^c u^d,  with h_{ab} = g_{ab} + u_a u_b

    Parameters
    ----------
    C : dict
        Covariant Weyl components.
    metric : MetricTensor
    u_cov : list/Matrix (length 4)
        Covariant components u_a (we'll raise to get u^a).

    Returns
    -------
    dict
        {'U': sympy expr, 'P_cov': Matrix(4,1), 'P_contra': Matrix(4,1)}
        where P is orthogonal to u (u^a P_a = 0) up to algebraic simplification.
    """
    g = Matrix(metric.tensor())
    g_inv = g.inv()
    n = g.shape[0]

    # Build Bel–Robinson tensor
    T = bel_robinson_from_weyl(C, metric)

    u_cov = Matrix(u_cov)
    u_up = g_inv * u_cov  # u^a

    # U = T_abcd u^a u^b u^c u^d
    U = 0
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    U += T[(a, b, c, d)] * u_up[a] * u_up[b] * u_up[c] * u_up[d]
    U = sp.simplify(U)

    # Projector h_a^e = delta_a^e + u_a u^e
    delta = sp.eye(n)
    h_mixed = delta.copy()
    for a in range(n):
        for e in range(n):
            h_mixed[a, e] = sp.simplify(delta[a, e] + u_cov[a] * u_up[e])

    # P_a = - h_a^e T_ebcd u^b u^c u^d
    P_cov = sp.MutableDenseMatrix(n, 1, [0]*n)
    for a in range(n):
        s = 0
        for e in range(n):
            # S_e := T_ebcd u^b u^c u^d
            S_e = 0
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        S_e += T[(e, b, c, d)] * u_up[b] * u_up[c] * u_up[d]
            s += h_mixed[a, e] * S_e
        P_cov[a, 0] = sp.simplify(-s)

    # Raise index for convenience
    P_contra = sp.simplify(g_inv * P_cov)

    if show_results:
        display(Math(r"\mathcal{U} = " + sp.latex(U)))
        display(Math(r"\mathcal{P}^a = " + sp.latex(P_contra)))

    return {"U": U, "P_cov": P_cov, "P_contra": P_contra}

def project_flux_observer(P_contra, u_contra, metric, show_results=True):
    """
    Project the super-Poynting vector P^a into the local orthonormal spatial triad
    of a given observer (u^a).

    Returns P^{(r)}, P^{(theta)}, P^{(phi)} in the observer's rest frame.
    """
    g = Matrix(metric.tensor())
    g_inv = g.inv()

    # --- Orthonormal triad (assuming spherical coords, circular motion) ---
    e_r = Matrix([0, 1/sp.sqrt(g[1,1]), 0, 0])
    e_th = Matrix([0, 0, 1/sp.sqrt(g[2,2]), 0])
    phi_basis = Matrix([0, 0, 0, 1])

    # Orthogonalize phi-basis to u
    u_cov = g * u_contra
    u_dot_phi = (u_cov.T * phi_basis)[0]
    e_phi_tilde = phi_basis - u_dot_phi * u_contra
    norm_phi = sp.sqrt((g * e_phi_tilde).dot(e_phi_tilde))
    e_phi = sp.simplify(e_phi_tilde / norm_phi)

    # --- Components in the local triad ---
    P_r = sp.simplify((g * e_r).dot(P_contra))
    P_th = sp.simplify((g * e_th).dot(P_contra))
    P_ph = sp.simplify((g * e_phi).dot(P_contra))

    if show_results:
        display(Math(
            r"\mathcal{P}^{(\hat{r})} = " + sp.latex(P_r) + r", \quad " +
            r"\mathcal{P}^{(\hat{\theta})} = " + sp.latex(P_th) + r", \quad " +
            r"\mathcal{P}^{(\hat{\phi})} = " + sp.latex(P_ph)
        ))

    return {"P^r_hat": P_r, "P^θ_hat": P_th, "P^φ_hat": P_ph}


# ---------- Newman–Penrose scalars from a null tetrad ----------
from IPython.display import display, Math

def np_scalars_from_weyl(C, tetrad, show_results=True):
    """
    Compute Newman–Penrose Weyl scalars Ψ0–Ψ4 from a given null tetrad.

    Parameters
    ----------
    C : dict
        Weyl tensor components {(a,b,c,d): value}.
    tetrad : dict
        Keys 'l','n','m','mbar' each contain a 4-vector (contravariant components).
    show_results : bool, optional
        If True, display the results in LaTeX.

    Returns
    -------
    dict with {'Psi0', 'Psi1', 'Psi2', 'Psi3', 'Psi4'}
    """

    l = Matrix(tetrad['l'])
    n = Matrix(tetrad['n'])
    m = Matrix(tetrad['m'])
    mb = Matrix(tetrad['mbar'])

    def C_contract(v1, v2, v3, v4):
        s = 0
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        s += C[(a,b,c,d)] * v1[a]*v2[b]*v3[c]*v4[d]
        return sp.simplify(s)

    Psi0 = C_contract(l, m, l, m)
    Psi1 = C_contract(l, n, l, m)
    Psi2 = C_contract(l, m, mb, n)
    Psi3 = C_contract(l, n, mb, n)
    Psi4 = C_contract(mb, n, mb, n)

    if show_results:
        display(Math(
            r"\Psi_0 = " + sp.latex(Psi0) + r",\quad " +
            r"\Psi_1 = " + sp.latex(Psi1) + r",\quad " +
            r"\Psi_2 = " + sp.latex(Psi2) + r",\quad " +
            r"\Psi_3 = " + sp.latex(Psi3) + r",\quad " +
            r"\Psi_4 = " + sp.latex(Psi4)
        ))

    return {
        "Psi0": Psi0,
        "Psi1": Psi1,
        "Psi2": Psi2,
        "Psi3": Psi3,
        "Psi4": Psi4,
    }

