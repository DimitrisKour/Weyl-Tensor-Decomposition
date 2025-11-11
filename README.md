## Project Overview

This repository accompanies the presentation **_Tidal Fields and Gravitational Waves: A Weyl Tensor Perspective_**.  
It provides a compact computational framework for studying the **electric–magnetic (E/B) decomposition of the Weyl tensor**,  
the associated curvature invariants *(J₁, J₂)*, and their relation to gravitational radiation across several spacetime models.

The notebooks combine **symbolic** and **numerical** approaches using `SymPy`, `EinsteinPy`, and `NumPy` to explore:
- The definition and physical meaning of **tidal fields** (E, B tensors)
- The behaviour of curvature invariants in Schwarzschild and Kerr geometries
- The identification of **radiative degrees of freedom** via \( \psi_4 \sim \ddot{h}_+ - i\,\ddot{h}_\times \)
- Fourier-domain representations of plane and binary gravitational waves

Each example builds conceptually upon the previous one:
1. **Conformally flat spacetime** – verifying the vanishing of curvature invariants  
2. **Schwarzschild** – symbolic computation of E, B, ψ scalars for static/circular observers  
3. **Kerr** – numerical evaluation of E, B, J₁, J₂ for static and rotating observers  
4. **Plane GW on Minkowski** – ψ₄ equivalence and monochromatic FFT  
5. **Quasi-circular binary (far zone)** – waveform analysis and frequency content

Together, these modules provide a unified computational lens on how **curvature, tidal fields, and radiation** emerge from the geometry of spacetime.
