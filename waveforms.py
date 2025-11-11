# waveforms.py
# Utilities for GW-like waveforms: symbolic-to-numeric conversion, plotting, FFT.
# Assumes SI units throughout:
#   - time t in seconds
#   - frequency in Hz (auto-scales to mHz/kHz just for labeling)
#   - strain h is dimensionless

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# ---------- Physical parameter builders (SI) ----------

def derived_binary_params(G, c, m1, m2, r0, r_det, calI, phi0, fgw=None):
    """
    Build a consistent parameter dict from *physical* inputs.

    Inputs (SI):
      G, c        : constants
      m1, m2      : component masses [kg]
      r0          : orbital separation [m]
      r_det       : distance detector–source [m]
      calI        : inclination 𝓲 [rad]
      phi0        : phase offset ϕ0 [rad]
      fgw         : optional GW frequency [Hz]; if None, set from Kepler

    Returns dict you can use for substitution/lambdify.
    """
    M  = m1 + m2
    mu = m1*m2 / M
    Mchirp = (mu**3 * M**2)**(1/5)

    if fgw is None:
        # Keplerian Ω^2 = GM/r0^3, and f_gw = Ω/π
        Omega = np.sqrt(G*M / r0**3)
        fgw = Omega / np.pi

    return {
        "G": G, "c": c,
        "m1": m1, "m2": m2, "M": M, "mu": mu, "M_chirp": Mchirp,
        "r0": r0, "r": r_det, "f_gw": fgw,
        "calI": calI, "phi0": phi0,
    }

def binary_strain_symbolic(t):
    """
    Maggiore-leading-order monochromatic strains (symbolic), evaluated at t (you can later t->t - r/c).

    h_+(t) = A * (1+cos^2 𝓲)/2 * cos(2π f_gw t + 2 φ0)
    h_×(t) = A * cos 𝓲         * sin(2π f_gw t + 2 φ0)

    with A = 4/r * (G M_chirp / c^2)^{5/3} * (π f_gw / c)^{2/3}
    """
    G, c = sp.symbols("G c", positive=True, real=True)
    M_chirp, f_gw = sp.symbols("M_chirp f_gw", positive=True, real=True)
    r, calI, phi0 = sp.symbols("r calI phi0", real=True)
    A = 4/r * (G*M_chirp/c**2)**sp.Rational(5,3) * (sp.pi*f_gw/c)**sp.Rational(2,3)
    phase = 2*sp.pi*f_gw*t + 2*phi0
    hp = A * (1+sp.cos(calI)**2)/2 * sp.cos(phase)
    hx = A * sp.cos(calI)          * sp.sin(phase)
    return hp, hx, {"G":G,"c":c,"M_chirp":M_chirp,"f_gw":f_gw,"r":r,"calI":calI,"phi0":phi0}

def brinkmann_waveform_symbolic(u=None):
    """
    Symbolic generator for monochromatic Brinkmann plane waves:

        h_+(u) = h0 * cos(ω u)
        h_×(u) = ε * h0 * sin(ω u)

    Returns
    -------
    h_plus_sym, h_cross_sym, symmap : (sympy.Expr, sympy.Expr, dict)
        Symbolic expressions and a dictionary of parameter symbols.
    """
    # --- symbols ---
    if u is None:
        u = sp.Symbol("u", real=True)
    h0, omega, eps = sp.symbols("h0 omega eps", real=True)
    # --- symbolic forms ---
    h_plus = h0 * sp.cos(omega * u)
    h_cross = eps * h0 * sp.sin(omega * u)
    symmap = {"h0": h0, "omega": omega, "eps": eps, "u": u}
    return h_plus, h_cross, symmap

def make_brinkmann_waveform(h0=1e-21, f_hz=1.0, eps=0.5):
    """Return callables hp(u), hx(u) for a Brinkmann plane wave."""
    u = sp.Symbol("u", real=True)
    hp_sym, hx_sym, symmap = brinkmann_waveform_symbolic(u)
    params = {
        symmap["h0"]: h0,
        symmap["omega"]: 2*np.pi*f_hz,
        symmap["eps"]: eps
    }
    hp, hx = lambdify_waveforms(hp_sym, hx_sym, u, params)
    return hp, hx, f_hz

# ---------- Time utilities ----------

def make_time_array(duration_s: float, fs_hz: float, t0: float = 0.0):
    """
    Build a uniform time array.
    duration_s : total duration in seconds
    fs_hz      : sampling rate in Hz
    t0         : start time (s)
    Returns: t (N,), dt, fs, N
    """
    N = int(np.round(duration_s * fs_hz))
    if N < 2:
        raise ValueError("Need at least 2 samples.")
    t = t0 + np.arange(N) / fs_hz
    dt = 1.0 / fs_hz
    return t, dt, fs_hz, N

# ---------- Symbolic → numeric helpers ----------

def lambdify_waveforms(hp_sym, hx_sym, t_sym, param_values: dict, modules="numpy"):
    hp_eval = sp.simplify(hp_sym.subs(param_values))
    hx_eval = sp.simplify(hx_sym.subs(param_values))
    hp = sp.lambdify(t_sym, hp_eval, modules)
    hx = sp.lambdify(t_sym, hx_eval, modules)
    return hp, hx

def resolve_params(symmap: dict, phys: dict):
    """
    Build a Symbol->numeric dict using the symbol objects in `symmap`
    and the numeric values in `phys`. Assumes:
      - symmap has SYMBOLS as values, e.g. {"G": G, "c": c, ...}
      - phys has NUMBERS keyed by the same names, e.g. {"G": 6.67e-11, "c": 299792458, ...}
    """
    out = {}
    for name, sym in symmap.items():
        if name in phys:
            out[sym] = phys[name]
    return out

def psi4_from_h_symbolic(hp_sym, hx_sym, t_sym):
    """
    Build symbolic psi4(t) from h_+(t), h_x(t):
        psi4 = -1/2 (h_plus'' - i h_cross'')
    Returns: psi4_sym
    """
    psi4_sym = -sp.Rational(1,2) * (sp.diff(hp_sym, t_sym, 2) - sp.I*sp.diff(hx_sym, t_sym, 2))
    return sp.simplify(psi4_sym)

def substitute_and_lambdify(expr_sym, t_sym, param_values: dict, modules="numpy"):
    """
    Generic substitute and lambdify for any symbolic expression of t.
    """
    expr_eval = sp.simplify(expr_sym.subs(param_values))
    return sp.lambdify(t_sym, expr_eval, modules)

def _window_array(N, window):
    if window is None:
        return np.ones(N)
    if isinstance(window, str):
        if window.lower() in ("hann","hanning"):
            return np.hanning(N)
        elif window.lower() == "boxcar":
            return np.ones(N)
        else:
            raise ValueError("Unsupported window type.")
    if callable(window):
        return window(N)
    w = np.asarray(window)
    if w.size != N:
        raise ValueError("Window length mismatch.")
    return w

def amplitude_spectrum(x, fs, window="hann", pad_to_pow2=True):
    """Dimensionless single-sided amplitude spectrum."""
    x = np.asarray(x)
    N = x.size
    w = _window_array(N, window)
    cg = w.mean()
    xw = x * w
    N_fft = 1 << (N - 1).bit_length() if pad_to_pow2 else N
    X = np.fft.rfft(xw, n=N_fft)
    f = np.fft.rfftfreq(N_fft, d=1.0/fs)
    A = (2.0 / (N * cg)) * np.abs(X)
    return f, A

def asd_one_sided(x, fs, window="hann", pad_to_pow2=True):
    """One-sided amplitude spectral density (strain/√Hz)."""
    x = np.asarray(x)
    N = x.size
    w = _window_array(N, window)
    xw = x * w
    W2 = np.dot(w, w)
    N_fft = 1 << (N - 1).bit_length() if pad_to_pow2 else N
    X = np.fft.rfft(xw, n=N_fft)
    f = np.fft.rfftfreq(N_fft, d=1.0/fs)
    psd_two = (1.0 / (fs * W2)) * (np.abs(X)**2)
    psd_one = psd_two.copy()
    if N_fft > 1:
        psd_one[1:-1] *= 2.0
    asd = np.sqrt(psd_one)
    return f, asd

def plot_spectrum(t, x, mode="amplitude", window="hann", pad_to_pow2=True, title=None):
    """Plot amplitude spectrum (dimensionless) or ASD (strain/√Hz)."""
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    if callable(x):
        x = x(t)
    if mode == "amplitude":
        f, Y = amplitude_spectrum(x, fs, window, pad_to_pow2)
        ylab = "single-sided amplitude"
    elif mode == "asd":
        f, Y = asd_one_sided(x, fs, window, pad_to_pow2)
        ylab = r"ASD [strain / $\sqrt{\mathrm{Hz}}$]"
    else:
        raise ValueError("mode must be 'amplitude' or 'asd'")
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
    ax.plot(f, Y)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    if title: ax.set_title(title)
    return fig, ax



# ---------- Plotting ----------

def plot_hplus_hcross(t, hp, hx, title=None):
    """
    Two-panel figure: h_plus(t) and h_cross(t).
    hp, hx can be arrays (preferred) or callables (then they are sampled on t).
    """
    if callable(hp):
        hp = hp(t)
    if callable(hx):
        hx = hx(t)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
    axes[0].plot(t, hp, color="black")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel(r"$h_+(t)$")

    axes[1].plot(t, hx, color="black")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel(r"$h_\times(t)$")

    if title:
        fig.suptitle(title, y=0.89, fontsize=19)
    return fig, axes

def plot_strain_magnitude(t, hp, hx, title=None, label=r"$|h(t)|$"):
    """
    Single-panel figure for |h(t)| = sqrt(hp^2 + hx^2).
    hp, hx may be arrays or callables on t.
    """
    if callable(hp):
        hp = hp(t)
    if callable(hx):
        hx = hx(t)
    hmag = np.sqrt(hp**2 + hx**2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
    ax.plot(t, hmag, color="black")
    ax.set_xlabel("t [s]")
    ax.set_ylabel(label)
    if title:
        ax.set_title(title)
    return fig, ax

# ---------- FFT utilities ----------

def _coherent_gain(window):
    """
    Coherent gain of the window for amplitude correction (preserve sinus amplitude).
    CG = (1/N) * sum(w[n])
    """
    return window.mean()



def fft_of_strain(t, hp, hx=None, which="hplus", window="hann", pad_to_pow2=True):
    """
    FFT of a selected strain channel:
        which="hplus" -> FFT of h_+(t)
        which="hcross"-> FFT of h_×(t)
        which="hmag"  -> FFT of |h(t)|

    hp, hx may be arrays or callables on t.
    Returns: (f_scaled, A, unit_label, f_hz_raw)
    """
    if callable(hp):
        hp = hp(t)
    if hx is not None and callable(hx):
        hx = hx(t)

    if which == "hplus":
        x = hp
        ylab = r"$|H_+(f)|$"
    elif which == "hcross":
        if hx is None:
            raise ValueError("Need hx for which='hcross'.")
        x = hx
        ylab = r"$|H_\times(f)|$"
    elif which == "hmag":
        if hx is None:
            raise ValueError("Need hx for which='hmag'.")
        x = np.sqrt(hp**2 + hx**2)
        ylab = r"$|H(f)|$"
    else:
        raise ValueError("which must be 'hplus', 'hcross', or 'hmag'.")

    # Sampling rate from t:
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    f, A = _one_sided_amplitude_spectrum(x, fs, window=window, pad_to_pow2=pad_to_pow2)
    f_scaled, unit = _scale_frequency_axis(f)
    return f_scaled, A, unit, f

def auto_freq_scale(fgw):
    """
    Choose frequency scaling and label based on f_gw.
    Returns (scale_factor, unit_label), where:
        f_plot = f_true * scale_factor
    """
    if fgw < 1e-3:
        return 1e3, "mHz"
    elif fgw < 1:
        return 1, "Hz"
    elif fgw < 1e3:
        return 1e-3, "kHz"
    else:
        return 1e-6, "MHz"

def plot_fft(t, hp, hx=None, which="hplus", window="hann", pad_to_pow2=True, xlim=None, ylim=None, title=None):
    """
    Convenience function: compute and plot FFT for selected channel.
    """
    f_scaled, A, unit, _ = fft_of_strain(t, hp, hx, which=which, window=window, pad_to_pow2=pad_to_pow2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.2), constrained_layout=True)
    ax.plot(f_scaled, A, color="black")
    ax.set_xlabel(f"frequency [{unit}]")
    ax.set_ylabel(r"single-sided amplitude")
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    if title: ax.set_title(title)
    return fig, ax

def plot_fft_waveform(
    t, hp, hx=None, f_gw=None, title=None,
    figsize=(6,4), window=True, pad_factor=8
):
    """
    Compute and plot |h̃(f)| for h₊(t) (and optionally h×(t)).

    Parameters
    ----------
    t : array
        Time array [s].
    hp, hx : array-like or callable
        Waveforms; if callable, evaluated on t.
    f_gw : float, optional
        Characteristic GW frequency [Hz] (for scaling + axis window).
    title : str, optional
        Figure title.
    figsize : tuple
        Figure size.
    window : bool
        Apply Hann window before FFT to reduce edge leakage.
    pad_factor : int
        Zero-padding factor (e.g. 8 -> pad signal to 8× its length).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Evaluate callables ---
    if callable(hp): hp = np.asarray(hp(t), dtype=float)
    if hx is not None and callable(hx): hx = np.asarray(hx(t), dtype=float)

    dt = t[1] - t[0]
    N = len(t)

    # --- Optional Hann window ---
    if window:
        win = np.hanning(N)
        hp = hp * win
        if hx is not None:
            hx = hx * win

    # --- Zero-padding for smoother spectrum ---
    Npad = pad_factor * N if pad_factor > 1 else N

    # --- FFT and amplitude spectrum ---
    freqs = np.fft.rfftfreq(Npad, dt)
    Hplus = np.fft.rfft(hp, n=Npad)
    amp = np.abs(Hplus)
    if hx is not None:
        Hcross = np.fft.rfft(hx, n=Npad)
        amp = np.sqrt(amp**2 + np.abs(Hcross)**2)

    # --- Physical frequency scaling ---
    scale, unit = auto_freq_scale(f_gw if f_gw else freqs[np.argmax(amp)])
    f_plot = freqs * scale
    f_peak = (f_gw or freqs[np.argmax(amp)]) * scale

    # --- Axis limits around f_gw ---
    fmin = 0
    fmax = 5 * (f_gw * scale if f_gw else f_peak * scale)

    # --- Plot ---
    plt.figure(figsize=figsize)
    plt.plot(f_plot, amp, 'k')
    plt.axvline(f_peak, color='red', ls='--', lw=0.8,
                label=f"$f_{{gw}}$ = {f_peak:.3g} {unit}")
    ax = plt.gca()
    ax.set_axisbelow(True)  # ensures grid is behind legend

    plt.xlim(fmin, fmax)
    plt.xlabel(f"frequency [{unit}]")
    plt.ylabel(r"$|\tilde{h}(f)|$")
    if title:
        plt.title(title)

    leg = plt.legend(
    frameon=True,
    fancybox=True,
    framealpha=1,
    edgecolor="#444444",
    facecolor="#ffffff",
    borderpad=0.6,
    fontsize=11.7
    )
    leg.set_zorder(1000)     # ensures legend on top

    plt.tight_layout()