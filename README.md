# me700-final
[![python](https://img.shields.io/badge/python-3.13.3-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![codecov](https://codecov.io/gh/rtengle/me700-final/graph/badge.svg?token=2Q1GLARIMP)](https://codecov.io/gh/rtengle/me700-final)
[![tests](https://github.com/rtengle/me700-final/actions/workflows/tests.yml/badge.svg)](https://github.com/rtengle/me700-final/actions)

## Overview

This codebase simulates the surface evolution of a temperature-controlled liquid space telescope using FEniCSx. In this system, a thin film of fluid sits atop a solid spherical surface with a prescribed base temperature. The top surface is free to deform and points into the deep vacuum of space where energy is released via radiation. 

The dynamics of this system is largely governed by two mechanisms: Surface tension where curvature along the fluid surface generates a surface pressure; and themocapillary flow where temperature gradients along a surface generates shear flow, draining the thin film from hot spots and into cold spots. With these effects the surface can be manipulated by prescribing a temperature profile at the solid surface the film sits atop. For the liquid space telescope, the following non-dimensional equation describes the time evolution of the fluid surface:

```math
\frac{\partial H}{\partial \tau} + \nabla \cdot \left( \frac{1}{3} S H^3 \nabla(\nabla^2 H) + \frac{1}{2} H^2 \nabla \theta_H \right) = 0 \\

\theta_H = \theta_0 - \frac{F}{K} H
```

where $H$ describes the fluid thickness and $\theta$ describes the fluid temperature. While steady-state solutions can be easily derived for a given fluid, not much is known about the evolution of a fluid surface shape including how stable these steady-states are. 

## Usage and Parameters

To set up the environment, run the following commands:

```
module load miniconda
mamba create -n fenicsx-env
mamba activate fenicsx-env
mamba install -c conda-forge fenics-dolfinx mpich pyvista
pip install -e .
```

With the environment set up, call ```run_sim(params)``` with ```params``` being a dictionary containing the following parameters:

```
gamma0 : Polar dista covered by the telescope surface
theta0 : Function describing the temperature profile
S : Surface tension constant
F/K : Ratio between the view factor F and the non-dimensional conductivity K
minsize : Minimum element size of the mesh
maxsize : Maximum element size of the mesh
Hpin : Pinned height of the fluid surface at the edge
etapin : Laplacian of the fluid surface at the edge
H0 : Starting fluid height
dt : Time step size
N : Number of steps
degree : Element degree
rtol : Relative tolerance of solver
foldername : Folder name for results data
filename : File name for results data
plot : Boolean for if a gif is produced
figurename : Name of gif file
```

The method ```run_sim``` is found in ```run_sim.py```. 
