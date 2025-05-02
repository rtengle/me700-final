from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
from dolfinx.plot import vtk_mesh
from dolfinx import fem, default_scalar_type, default_real_type, mesh
from basix.ufl import element, mixed_element
import gmsh
import pyvista
import numpy as np
import ufl
from petsc4py import PETSc

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from meshing import *
from weak_form import *

params = dict()
params['gamma0'] = 9 * np.pi/180
params['minsize'] = 0.01
params['maxsize'] = 0.01
params['dt'] = 0.01

model = create_gmsh(params)
solver, u, u0 = create_solver(params, model)

u0.x.array[:] = u.x.array

r = solver.solve(u)

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = u.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()

pass