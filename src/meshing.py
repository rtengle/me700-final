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
import sys

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

def create_mesh(params) -> tuple:
    """Creates the mesh for the fluid surface simulation. This is a flat 2D circle.

    Parameters
    ----------
    params : dict
        Dictionary containing all user-defined settings for the test. Needs to contain at least the following:
        gamma0 : float
            Half-arc length of analyzed region
        minsize : float
            Minimum element size
        maxsize : float
            Maximum element size

    Returns
    -------
    mesh_triplet : tuple
        Triplet containing the following:
        domain
            DolfinX mesh object
        cell_markers
            Topological domain tags for the mesh
        facet_markers
            Topological facet tags for the mesh
    """
    # Starts gmsh CAD kernel
    gmsh.clear()
    gmsh.initialize()

    # Our domain is a flat 2D disk. This also generates an exterior loop around it
    gmsh.model.occ.addDisk(0, 0, 0, params['gamma0'], params['gamma0'], 1)

    # We then synchronize the CAD model with our gmsh model allowing us to define our groups
    gmsh.model.occ.synchronize()
    
    # We now group together our fluid surface so that the mesh will build and the edge so that it's a facet

    # Groups together fluid surface
    gmsh.model.addPhysicalGroup(2, [1], 1, name='domain')
    # Groups together fluid surface boundary
    gmsh.model.addPhysicalGroup(1, [1], 1, name='edge')

    # We generate the mesh

    # Sets the mesh sizing
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", params['minsize'])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", params['maxsize'])
    # Generates mesh
    gmsh.model.mesh.generate(2)

    # Then conert it to a FEniCSx mesh

    # Converts gmsh model to a fenicsx mesh
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh_triplet = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)
    # Triplet contains: mesh object, cell tags, and facet tags

    return mesh_triplet