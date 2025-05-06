# me700-final

## Background

This project is a finite element simulation of a thin fluid film deforming under thermocapillary shaping. The general setup is a thin film sitting atop a solid surface. This surface is maintained at a specific temperature $\theta_0$. This produces a temperature gradient at the surface, driving fluid flow. This drains fluid from hot spots and builds it up at cold spots. This combined with capillary pressure can be used to shape fluid surfaces and the dynamics are described by the following equation:

```math
\frac{\partial H}{\partial \tau} + \nabla \cdot \left( \frac{1}{3} S H^3 \nabla(\nabla^2 H) + \frac{1}{2} H^2 \nabla \theta_H \right) = 0 \\

\theta_H = \theta_0 - \frac{F}{K} H
```

where $H$ describes the fluid thickness and $\theta$ describes the fluid temperature. While steady-state solutions can be easily derived for a given fluid, not much is known about the evolution of a fluid surface shape including how stable these steady-states are. 

## Overview

This codebase simulates a fluid film along a given surface. All that is needed to be done is use the ```run_sim``` command in the file of the same name and pass through a dict with the simulation parameters.