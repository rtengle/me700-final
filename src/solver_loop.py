import os

from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
from dolfinx.plot import vtk_mesh
from dolfinx import fem, default_scalar_type, default_real_type, mesh, log
from basix.ufl import element, mixed_element
import gmsh
import pyvista
import numpy as np
import ufl
from petsc4py import PETSc
import matplotlib as mpl

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from meshing import *
from weak_form import *

log.set_output_file("log.txt")

params = dict()
params['gamma0'] = 9 * np.pi/180
params['minsize'] = 5e-3
params['maxsize'] = 5e-3
params['Hpin'] = 1
params['etapin'] = 0
params['H0'] = 1
params['dt'] = 1e-4
params['S'] = 1e-1
params['N'] = 50

def theta(x):
    return 3*ufl.exp(-100*x[0]**2 - 100*x[1]**2)

mesh_triplet = create_mesh(params)
domain, cell_markers, facet_markers = mesh_triplet

solver, u, u0, S = create_solver(params, mesh_triplet, theta)

file = XDMFFile(MPI.COMM_WORLD, "results/output.xdmf", "w")
file.write_mesh(domain)

V0, dofs = S.sub(0).collapse()

h = u.sub(0)



pyvista.OFF_SCREEN = True

pyvista.start_xvfb()
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Create a plotter that will create a .gif
plotter = pyvista.Plotter()
plotter.open_gif("figures/Hred_time.gif", fps=10)
Nslides = params['N']

# Stores the H data
grid.point_data["H"] = u.x.array[dofs]
# I think warp adds a height map on a grid
warped = grid.warp_by_scalar("H", factor=1)

# Set the color map
viridis = mpl.colormaps.get_cmap("magma").resampled(25)

# This is the arguments for the color bar
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

# Adds in the mesh with a color bar and height
renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                            cmap=viridis, scalar_bar_args=sargs,
                            clim=[0, 2*max(u.x.array[dofs])])

t = 0
for i in range(params['N']):
    t += params['dt']
    u0.x.array[:] = u.x.array
    r = solver.solve(u)
    print(f"Step {i}: num iterations: {r[0]}")
    file.write_function(h, t)

    # Replaces the height map from before
    new_warped = grid.warp_by_scalar("H", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["H"][:] = u.x.array[dofs]
    # Writes the plot to the next frame
    plotter.write_frame()

plotter.close()

file.close()

# pyvista

u_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
u_grid.point_data["u"] = u.x.array[dofs].real
u_grid.set_active_scalars("u")

warped = u_grid.warp_by_scalar()

plotter = pyvista.Plotter()
plotter.add_mesh(warped, show_edges=True)
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("figures/fundamentals_mesh.png")

pass