# FRG

Package for Functional Renormalisation Group calculations within the next-to-leading order (NLO) approximation. 

## About the FRG

In a nutshell, the Renormalisation Group method is a smart coarse-graining: it tells how a system's effective description changes with scale.

The FRG method [1] is used and shows impressive results in various domains of physics: equilibrium and out-of-equilibrium phenomena, quantum and classical, from condensed matter to quantum gravity. Here, we focus on classical out-of-equilibrium physics: stochastic Navier-Stokes equation, Kardar-Parisi-Zhang equation, while the calculations for other models (whatever physics they describe) can be implemented in the same manner, which was the motivation of creating this repository.

Solution of the FRG equation - the Wetterich equation - is possible under some approximation. In this repository we implement the NLO approximation, which is very expressive and captures well the two-point correlations.

On top of that, we reinforce this method with the *two grids* scheme [2], which permits to obtain two-point correlation functions in a wide range of length- and time-scales.

## Files

Branch 'main' contains the implementation of the FRG equations solution for the Kardar-Parisi-Zhang model in arbitrary dimension.

- `flow_NLO.py`: the class FlowNLO for FRG equations solution 
    - the parameter `approximation` can be set to "NLO" or "LO";
    - the flowing functions are discretized on a log grids of frequency and momentum;
    - the evolution of the flowing coupling, anomalous dimension and functions is saved to corresponding files.
- `regulator.py`: collection of regulators and their derivatives that you can plug into the flow
- `my_plot_NLO.py`: some frequently-used plotting routines to visualize the FRG results

Usage examples:
- `NLO_...ipynb`:  an example of the flow within the NLO approximation for the KPZ equation in 2D.
- `LO_...ipynb`:  an example of the flow within the LO approximation and same model.

## Upcoming

dev: making a package for FRG calculations that will be easily-expandable to different models.

---------

#### Author
Liubov Gosteva

#### Licence
See LICENCE.md

#### References
[1] N. Dupuis, L. Canet, A. Eichhorn, W. Metzner, J.M. Pawlowski, M. Tissier, N. Wschebor, The nonperturbative functional renormalization group and its applications,
Physics Reports, V.910, p. 1-114 (2021), https://doi.org/10.1016/j.physrep.2021.01.001.

[2] L.Gosteva, N. Wschebor, L. Canet, Unveiling the different scaling regimes of the one-dimensional Kardar–Parisi–Zhang–Burgers equation using the functional renormalisation group, J. Stat. Mech., p. 114002 (2025), https://doi.org/10.1088/1742-5468/ae1b85.