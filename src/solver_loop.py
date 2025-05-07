import os

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
import matplotlib as mpl

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from meshing import *
from weak_form import *

def configure_gif_plotter(params, warped, plotter):
    # Set the color map
    cmap = mpl.colormaps.get_cmap("magma").resampled(50)

    # This is the arguments for the color bar
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    # Adds in the mesh with a color bar and height
    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False, cmap=cmap, scalar_bar_args=sargs, clim=[0, 2])

    return renderer

def update_warped(warped, grid, plotter, sx):
    # Replaces the height map from before
    new_warped = grid.warp_by_scalar("H", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["H"][:] = sx
    # Writes the plot to the next frame
    plotter.write_frame()

def solver_loop(params, mesh_triplet, solver, function_triplet):
    domain, cell_markers, facet_markers = mesh_triplet
    s, s0, S = function_triplet
    
    # file = XDMFFile(MPI.COMM_WORLD, "results/output.xdmf", "w")
    # file.write_mesh(domain)

    S0, dofs = S.sub(0).collapse()

    h = s.sub(0)

    if params['plot']:
        pyvista.OFF_SCREEN = True
        pyvista.start_xvfb()

        grid = pyvista.UnstructuredGrid(*vtk_mesh(S0))

        # Create a plotter that will create a .gif
        plotter = pyvista.Plotter()
        plotter.open_gif(f"figures/{params['figurename']}.gif", fps=10)

        # Stores the H data
        grid.point_data["H"] = s.x.array[dofs]
        warped = grid.warp_by_scalar("H", factor=1)
        # I think warp adds a height map on a grid

        renderer = configure_gif_plotter(params, warped, plotter)

    # with io.VTXWriter(domain.comm, f'{params['foldername']}/{params['filename']}.bp', [s.sub(0), s.sub(1)]) as vtx:
    t = 0
    # vtx.write(t)
    for i in range(params['N']):
        t += params['dt']
        s0.x.array[:] = s.x.array
        r = solver.solve(s)
        print(f"Step {i}: num iterations: {r[0]}")
        # vtx.write(t)
        if params['plot']:
            update_warped(warped, grid, plotter, s.x.array[dofs])

    if params['plot']:
        plotter.close()
    # vtx.close()