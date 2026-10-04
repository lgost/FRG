# FRG

Package for Functional Renormalisation Group calculations within the next-to-leading order (NLO) approximation and the *two grids* scheme. 

## About the FRG

In a nutshell, the Renormalisation Group method is a smart coarse-graining: it tells how a system's effective description changes with scale.

The FRG method [1] is used and shows impressive results in various domains of physics: equilibrium and out-of-equilibrium phenomena, quantum and classical, from condensed matter to quantum gravity. Here, we focus on classical out-of-equilibrium physics: stochastic Navier-Stokes equation, Kardar-Parisi-Zhang equation, while the calculations for other models (whatever physics they describe) can be implemented in the same manner, which was the motivation of creating this repository.

Solution of the FRG equation - the Wetterich equation - is possible under some approximation. In this repository we implement the NLO approximation, which is very expressive and captures well the two-point correlations.

On top of that, we reinforce this method with the *two grids* scheme [2], which permits to obtain two-point correlation functions in a wide range of length- and time-scales.

## Files

<pre>
📂 FRG
├── 📁 flow
│   ├── 📁 models
│   │   ├── 📄 model_base.py   ← Base class for Model classes, that define a physical model and the approximation for the FRG equation. It contains methods for calculation of rhs of flow equations (which define the model, they are abstract, to be implemented in child classes), as well as computational details: regulators, grids, etc.
│   │   ├── 📄 model_kpz.py   ← Implementation for the KPZ equation.
│   │   ├── ...
│   │   └── 📄 model_ADD_YOUR_MODEL.py  
│   ├── 📁 evolution_two
│   │   ├── 📄 evolution_two_base.py   ← Base class for the dimensionful evolution. Records the IC (the correlation function) for the large-p equation.
│   │   ├── 📄 evo2_kpz.py   ← Implementation for the KPZ equation.
│   │   ├── ...
│   │   └── 📄 evo2_ADD_YOUR_MODEL.py  
│   ├── 📄 create.py   ← The "factory". Sets up a given evolution for a given model. 
│   ├── 📄 evolution.py   ← Class to integrate dimensionless flow equations. It owns the current state of the flowing variables.
│   ├── 📄 regulator.py   ← Collection of regulators to plug into the flow.
│   └── ...
└── 📁 usage_examples
    ├── 📄 KPZ_1D_twogrids.ipynb   ← KPZ equation in 1D with in the NLO approximation and recording of the dimensionful corr function with initial condition g_in=1. Result: KPZ scaling in IR and EW scaling in UV.
    └── 📄 KPZ_1D_twogrids_g300.ipynb   ← Same for g_in=300 (low viscosity). Result: KPZ scaling in IR and inviscid scaling in UV.
</pre>

The *two grids* scheme integrates the small-momentum flow equation (NLO or LO) on a dimensionless grid. The result is recorded at a certain RG time (a certain scale) and used as an initial condition for the large-momentum flow equations that are integrated on a dimensionful grid.

- Dimensionless flow `evolution.py`
    - the parameter `approximation` can be set to "NLO" or "LO";
    - the flowing functions are discretized on a log grids of frequency and momentum;
    - the evolution of the flowing coupling, anomalous dimension and flowing functions is saved to corresponding files.

- Dimensionful flow `evolution_two_base.py`
    - can be added on top of the dimensionless flow in the factory `create.py`;
    - records the IC for the large-p equation; the integration of the large-p equation is performed elsewhere.

## Upcoming

dev:

todo's: add Navier-Stokes model; add Crank–Nicolson algorithm to integrate the dimensionful flow.

---------

#### Author
Liubov Gosteva

#### Licence
See LICENCE.md

#### References
[1] N. Dupuis, L. Canet, A. Eichhorn, W. Metzner, J.M. Pawlowski, M. Tissier, N. Wschebor, The nonperturbative functional renormalization group and its applications,
Physics Reports, V.910, p. 1-114 (2021), https://doi.org/10.1016/j.physrep.2021.01.001.

[2] L.Gosteva, N. Wschebor, L. Canet, Unveiling the different scaling regimes of the one-dimensional Kardar–Parisi–Zhang–Burgers equation using the functional renormalisation group, J. Stat. Mech., p. 114002 (2025), https://doi.org/10.1088/1742-5468/ae1b85.