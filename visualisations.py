import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from plotting import rcparams  # your aesthetic settings file
import matplotlib.ticker as mticker
import seaborn as sns

# apply style globally
rcParams.update(rcparams)

# --- 1. Single scalar vs radius ---
def plot_scalar_vs_r(expr, r_sym, subs_dict=None, label=None, title=None,
                     orbit_type="stat", r_range=None, N=400):
    """
    Plot a scalar vs radius r, with automatic minimum r depending on orbit type.

    orbit_type:
      "stat" -> static observer: r_min = 2M
      "circ" -> circular orbit:  r_min = 6M
    """
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt

    subs_dict = subs_dict or {}

    # Extract M if present; default assume M=1
    Mval = subs_dict.get(sp.Symbol("M"), 1)

    if orbit_type == "stat":
        r_min = 2 * Mval
    elif orbit_type == "circ":
        r_min = 6 * Mval
    else:
        raise ValueError("orbit_type must be 'stat' or 'circ'")

    # If user did not give custom r_range, define one automatically
    if r_range is None:
        r_range = (r_min * 1.05, 15 * Mval)

    f = sp.lambdify(r_sym, expr.subs(subs_dict), "numpy")
    r_vals = np.linspace(float(r_range[0]), float(r_range[1]), N)
    y_vals = f(r_vals)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(r_vals, y_vals, color="black")
    ax.set_xlabel(r"$r/M$")
    ax.set_ylabel(label or r"$f(r)$")
    ax.set_title(title or "")
    plt.tight_layout()

# --- 2. Multiple scalars for comparison ---
def plot_compare_dual(exprs, labels, r_sym, subs_dict=None, title=None, r_range=(2.1, 10), N=400):
    """
    Plot two scalar quantities (e.g. |E|² and |B|²) vs r on separate axes (1x2 layout).
    """
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt

    subs_dict = subs_dict or {}
    r_vals = np.linspace(float(r_range[0]), float(r_range[1]), int(N))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    fig.suptitle(title or "", fontsize=18)

    for ax, expr, lab in zip(axes, exprs, labels):
        expr_sub = sp.simplify(expr.subs(subs_dict))
        f = sp.lambdify(r_sym, expr_sub, modules="numpy")

        y_raw = f(r_vals)
        y_vals = np.array(y_raw, dtype=float).ravel()
        y_vals = np.real_if_close(y_vals, tol=1e-12)

        # Safety: check shape and constancy
        if y_vals.ndim != 1 or y_vals.size != r_vals.size:
            print(f"⚠️  Warning: {lab} returned shape {y_vals.shape}, flattening to length {r_vals.size}")
            y_vals = np.resize(y_vals, r_vals.size)

        if np.allclose(y_vals, y_vals[0], atol=1e-12):
            print(f"⚠️  {lab} seems numerically constant over the sampled range!")

        # Clean non-finite
        mask = np.isfinite(y_vals)
        ax.plot(r_vals[mask], y_vals[mask], lw=2, label=lab)

        ax.set_xlabel(r"$r/M$")
        ax.set_ylabel(lab)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

def plot_compare_normalized_by_J1(E2_expr, B2_expr, J1_expr, r_sym,
                                  subs_dict=None, title=None,
                                  r_range=(2.1, 10), N=400, j1_eps=1e-14,
                                  J1_behavior="oscillating"):
    """
    Compare |E|² and |B|² normalized by |J₁| across r, showing the ratio B²/E².
    
    Parameters
    ----------
    E2_expr, B2_expr, J1_expr : sympy.Expr
        Sympy expressions for |E|², |B|², and J₁.
    r_sym : sympy.Symbol
        Radial variable.
    subs_dict : dict, optional
        Substitutions for parameters (e.g. {M:1, th:pi/2}).
    title : str, optional
        Figure title.
    r_range : tuple, optional
        Range of r values (start, end).
    N : int, optional
        Number of sample points.
    j1_eps : float, optional
        Small threshold to avoid dividing by ~0 J₁.
    J1_behavior : {"positive", "oscillating"}
        If "positive", only plot ratio (J₁ > 0 everywhere).
        If "oscillating", overlay dashed sign(J₁) on the same vertical scale.
    """
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    subs_dict = subs_dict or {}
    r_vals = np.linspace(float(r_range[0]), float(r_range[1]), int(N))

    # --- Evaluate expressions ---
    E2_l = sp.lambdify(r_sym, sp.simplify(E2_expr.subs(subs_dict)), "numpy")
    B2_l = sp.lambdify(r_sym, sp.simplify(B2_expr.subs(subs_dict)), "numpy")
    J1_l = sp.lambdify(r_sym, sp.simplify(J1_expr.subs(subs_dict)), "numpy")

    E2 = np.real_if_close(np.array(E2_l(r_vals), float).ravel(), tol=1e-12)
    B2 = np.real_if_close(np.array(B2_l(r_vals), float).ravel(), tol=1e-12)
    J1 = np.real_if_close(np.array(J1_l(r_vals), float).ravel(), tol=1e-12)

    good = np.isfinite(E2) & np.isfinite(B2) & np.isfinite(J1) & (np.abs(J1) > j1_eps)
    if not np.any(good):
        print("⚠️  No valid points after masking! Check range or j1_eps.")
        return

    r = r_vals[good]
    E2n = E2[good] / np.abs(J1[good])
    B2n = B2[good] / np.abs(J1[good])
    J1s = J1[good]
    ratio = np.where(E2n != 0, B2n / E2n, np.nan)

    # --- Colors ---
    cE = "#1F5673"  # deep steel blue
    cB = "#A67244"  # muted bronze
    cR = "black"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True)
    fig.suptitle(title or "Normalized tidal invariants", y=0.85, fontsize=18)

    # --- Left panel: normalized magnitudes ---
    ax0 = axes[0]
    ax0.plot(r, E2n, color=cE, label=r"$E^2/|J_1|$")
    ax0.plot(r, B2n, color=cB, label=r"$B^2/|J_1|$")
    ax0.set_xlabel(r"$r/M$")
    ax0.set_ylabel(r"Normalized magnitude")
    #ax0.set_yscale("symlog", linthresh=1e-6)
    ax0.set_yscale("linear")
    ax0.legend(frameon=False, loc="best")

    # --- Right panel: ratio ---
    ax1 = axes[1]
    ax1.plot(r, ratio, color=cR, label=r"$B^2/E^2$")
    ax1.set_xlabel(r"$r/M$")
    ax1.set_ylabel(r"Ratio $B^2/E^2$")
    ax1.set_yscale("symlog", linthresh=1e-6)

    if J1_behavior == "oscillating":
        # Symmetric tick layout (fixed)
        yticks = [-1e-2, -1e-1, -1e0, -1e-1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
        labels = [r"$-10^{0}$", r"$-10^{-1}$", r"$-10^{-2}$", r"$-10^{-3}$",
                  r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$"]
        ax1.yaxis.set_major_locator(mticker.FixedLocator(yticks))
        ax1.yaxis.set_major_formatter(mticker.FixedFormatter(labels))
        ax1.set_ylim(-1.2, 1.2)

        # Overlay sign(J1) (no centering manipulation)
        ax1b = ax1.twinx()
        ax1b.plot(r, np.sign(J1s), "k--", alpha=0.35, label="sign(J₁)")
        ax1b.set_ylabel("sign(J₁)")
        ax1b.set_ylim(ax1.get_ylim())  # share natural limits
        ax1b.set_yticks([-1, 0, 1])
        ax1b.set_yticklabels([r"$-1$", r"$0$", r"$+1$"])

        # Unified legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="best")

    else:
        # J1 positive: simpler ratio-only panel
        ax1.legend(frameon=False, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

# --- 3. Tensor heatmap (e.g. E_ab, B_ab) ---
def plot_tensor_matrix(T, subs_dict=None, r_val=6, title=None, cmap="crest"):
    """
    Visualize a tensor T (3×3 or 4×4) as a smooth heatmap at given r,
    with LaTeX-style indices (μ, ν) and a light, continuous color palette.

    Parameters
    ----------
    T : sympy.Matrix
        Tensor to visualize.
    subs_dict : dict, optional
        Substitutions for parameters (e.g. {M:1, th:pi/2}).
    r_val : float, optional
        Radial coordinate value for evaluation.
    title : str, optional
        Figure title.
    cmap : str, optional
        Colormap name. Supports both Seaborn and Matplotlib maps.
        Recommended: "crest" (softest), "mako", "rocket", "YlGnBu".
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    subs_dict = subs_dict or {}
    subs_dict = {**subs_dict, "r": r_val}

    # Evaluate tensor numerically
    T_eval = np.array(T.subs(subs_dict).evalf(), dtype=float)

    # Resolve colormap: try Seaborn first, else Matplotlib
    try:
        cmap_obj = sns.color_palette(cmap, as_cmap=True)
    except ValueError:
        cmap_obj = plt.get_cmap(cmap)

    # Create figure
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(T_eval, cmap=cmap_obj, interpolation="bilinear")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Component value", rotation=270, labelpad=15)

    # Determine tensor shape
    nrows, ncols = T_eval.shape

    # Axis labels and title
    ax.set_xlabel(r"Index $\nu$")
    ax.set_ylabel(r"Index $\mu$")
    ax.set_title(title or r"Tensor components $T^{\mu}{}_{\nu}$", pad=10)

    # LaTeX tick labels for indices
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels([fr"${i}$" for i in range(ncols)])
    ax.set_yticklabels([fr"${i}$" for i in range(nrows)])

    plt.tight_layout()

# --- 4. Vector component vs radius (e.g. P^r) ---
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def plot_vector_component(P, comp_idx, r_sym, subs_dict=None, title="", r_range=(2.1, 10.0), N=400):
    """
    Plot a chosen component P^a(r) of a 4-vector as function of r.
    comp_idx: 0=t, 1=r, 2=theta, 3=phi
    """
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt

    subs_dict = subs_dict or {}
    r_vals = np.linspace(float(r_range[0]), float(r_range[1]), int(N))

    # --- extract scalar component ---
    comp_expr = P[comp_idx, 0] if isinstance(P, sp.MatrixBase) else P[comp_idx]
    comp_expr = sp.simplify(comp_expr.subs(subs_dict))

    # --- evaluate safely ---
    f = sp.lambdify(r_sym, comp_expr, modules="numpy")
    y_raw = f(r_vals)

    # Flatten and clean any extra dimensions
    y_vals = np.array(y_raw, dtype=float).ravel()
    y_vals = np.real_if_close(y_vals, tol=1e-12)

    # In case of NaN/Inf near the horizon
    mask = np.isfinite(y_vals)
    r_plot = r_vals[mask]
    y_plot = y_vals[mask]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_plot, y_plot, lw=2)
    ax.set_xlabel(r"$r/M$")
    ax.set_ylabel(rf"$P^{{{comp_idx}}}$")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.5)