import os

from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
from dolfinx.plot import vtk_mesh
from dolfinx import fem, default_scalar_type, default_real_type, mesh, io
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

def add_colorbar(params, warped, plotter):
    """Internal function to configure gif plotter
    """
    # Set the color map
    cmap = mpl.colormaps.get_cmap("magma").resampled(50)

    # This is the arguments for the color bar
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    # Adds in the mesh with a color bar and height
    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False, cmap=cmap, scalar_bar_args=sargs, clim=[0, 2])

def update_warped(warped, grid, plotter, sx):
    """Internal function to update gif plot
    """
    # Replaces the height map from before
    new_warped = grid.warp_by_scalar("H", factor=1)
    warped.points[:, :] = new_warped.points
    warped.point_data["H"][:] = sx
    # Writes the plot to the next frame
    plotter.write_frame()

def solver_loop(params, mesh_triplet, solver, function_triplet):
    """Iterative loop that performs the implicit solver scheme for a series of timesteps

    Arguments
    ---------
        params : dict
        Dictionary containing all user-defined settings for the simulation. Needs to contain at least the following:
        plot : bool
            True/False for whether or not to plot the gif
        figurename : str
            Name of figure file
        foldername : str
            Name of folder the results are stored in
        filename : str
            Name of file the results are written to
        N : int
            Number of timesteps the analysis performs
        dt : float
            Time interval between steps
    """
    # Unpacks mesh and function triplets
    domain, cell_markers, facet_markers = mesh_triplet
    s, s0, S = function_triplet

    # Isolates the functionspace and node value locations (dofs) for H
    SH, dofs = S.sub(0).collapse()

    # pyvista configuration
    if params['plot']:
        # Sets pyvista to not plot on-screen and starts the virtual framebuffer
        pyvista.OFF_SCREEN = True
        pyvista.start_xvfb()

        # Gets an unstructured grid from the H function space
        grid = pyvista.UnstructuredGrid(*vtk_mesh(SH))

        # Create a plotter that will create a .gif
        plotter = pyvista.Plotter()
        plotter.open_gif(f"figures/{params['figurename']}.gif", fps=10)

        # Stores the H data
        grid.point_data["H"] = s.x.array[dofs]
        warped = grid.warp_by_scalar("H", factor=1)

        # Adds in the colorbar to the plot
        add_colorbar(params, warped, plotter)

    # Writes H and eta to file using vtx
    with io.VTXWriter(domain.comm, f'{params['foldername']}/{params['filename']}.bp', [s.sub(0), s.sub(1)]) as vtx:
        # Sets initial time
        t = 0
        # Writes the current H and eta at time 0
        vtx.write(t)
        # Time stepping loop
        for i in range(params['N']):
            # Advances time
            t += params['dt']
            # Sets previous surface to the current surface
            s0.x.array[:] = s.x.array
            # Solves for the new surface
            r = solver.solve(s)
            # Outputs where in the iteration process we are
            print(f"Step {i}: num iterations: {r[0]}")
            # Writes the current surface at time t
            vtx.write(t)
            # Updates the gif plot
            if params['plot']:
                update_warped(warped, grid, plotter, s.x.array[dofs])
        
        # Closes the plot
        if params['plot']:
            plotter.close()

        # Closes the vtx file
        vtx.close()