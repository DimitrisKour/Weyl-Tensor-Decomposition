# Tidal Fields and Gravitational Waves: A Weyl Tensor Perspective

This repository accompanies the presentation **_Tidal Fields and Gravitational Waves: A Weyl Tensor Perspective_**. It provides a compact computational framework for exploring the **electric–magnetic (E/B) decomposition of the Weyl tensor**, its associated **invariants (J₁, J₂)**, and their relation to **gravitational radiation** in both exact and approximate spacetimes.

## Overview

The project combines **symbolic** and **numerical** methods to study curvature and radiation within General Relativity. Starting from the Weyl tensor \( C_{\alpha\beta\gamma\delta} \), the code defines:

- **Electric and magnetic tidal tensors:**  
  \( E_{\alpha\beta} = C_{\alpha\mu\beta\nu} u^\mu u^\nu \),  
  \( B_{\alpha\beta} = \tfrac{1}{2}\,\epsilon_{\alpha\mu\sigma\tau} C^{\sigma\tau}{}_{\beta\nu} u^\mu u^\nu \)
- **Curvature invariants:**  
  \( J_1 = E^{ab}E_{ab} - B^{ab}B_{ab} \),  
  \( J_2 = 2 E^{ab}B_{ab} \)
- **Bel–Robinson energy quantities:** \( U, P_i \)
- **Newman–Penrose scalars:** \( \psi_0, \dots, \psi_4 \)

These quantities are used to interpret gravitational radiation in various contexts—from stationary geometries to plane and quasi-circular waves.

## Repository Structure

Weyl-Tensor-Decomposition/
├── Figures/                            # Output plots and figures
├── Theoretical Introduction.ipynb      # Markdown-based theoretical overview
├── Example 1 - Conformally Flat Spacetime.ipynb
│   → Symbolic check: Minkowski limit (E = B = J = 0)
├── Example 2 - Schwarzschild Spacetime.ipynb
│   → Symbolic computation of E, B, ψ scalars for static/circular observers
├── Example 3 - Kerr Spacetime.ipynb
│   → Numerical computation of E, B, J₁, J₂ for static and rotating observers
├── Example 4 - Plane GW on Minkowski.ipynb
│   → Symbolic derivation of ψ₄ ≈ ḧ₊ − i ḧ× and FFT of monochromatic wave
├── Example 5 - Quasi-Circular Binary in Far Zone.ipynb
│   → Time-series analysis and FFT for binary source waveforms
│
├── numerical.py                         # Numerical computation of E, B, J invariants
├── plotting.py                          # Aesthetic figure utilities and standardized plots
├── utils.py                             # Symbolic Weyl decomposition, ψ scalars, and tensor operations
├── visualisations.py                    # Symbolic visualization routines
├── waveforms.py                         # Time-domain → frequency-domain utilities (FFT, windowing)
└── requirements.txt                     # Python dependencies

## Installation

git clone https://github.com/DimitrisKour/Weyl-Tensor-Decomposition.git  
cd Weyl-Tensor-Decomposition  
pip install -r requirements.txt  

**Dependencies:** SymPy, NumPy, Matplotlib, EinsteinPy, SciPy

## Usage

1. Open any of the example notebooks in **Jupyter Lab** or **VS Code**.  
2. Run the cells sequentially to reproduce symbolic or numerical results.  
3. Figures are saved automatically in the `/Figures/` directory.

Example (symbolic use):

from utils import compute_weyl_decomposition  
E, B, J1, J2 = compute_weyl_decomposition(metric='Schwarzschild', observer='static')

Example (numerical use):

from numerical import compute_EB_J  
data = compute_EB_J(r_range=(2.1, 15), M=1.0, a=0.9, theta=np.pi/2)

## Contents Summary

| Section | Description |
|----------|-------------|
| **Theoretical Introduction** | Conceptual overview of the Weyl tensor, E/B decomposition, and gravitational radiation. |
| **Example 1** | Conformally flat (Minkowski) check — all invariants vanish. |
| **Example 2** | Schwarzschild spacetime: static vs circular observers. |
| **Example 3** | Kerr spacetime: numerical exploration of tidal invariants. |
| **Example 4** | Brinkmann plane wave on flat background; ψ₄ equivalence. |
| **Example 5** | Quasi-circular binary (far zone): waveform analysis and FFTs. |

## Reference Context

- S. Chandrasekhar, *The Mathematical Theory of Black Holes*  
- E. Poisson & C. Will, *Gravity: Newtonian, Post-Newtonian, Relativistic*  
- C.W. Misner, K.S. Thorne & J.A. Wheeler, *Gravitation*  
- G.F.R. Ellis & R. Maartens, *Relativistic Cosmology*  
- J. Stewart, *Advanced General Relativity*  
- M. Maggiore, *Gravitational Waves, Vol. 1*


## Author

**Dimitrios Kourtesis**  
MSc Subatomic Physics 
Greece
