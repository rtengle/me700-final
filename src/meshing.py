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

def create_gmsh(params):
    gmsh.initialize()
    surface = gmsh.model.occ.addDisk(0, 0, 0, params['gamma0'], params['gamma0'])

    gmsh.model.occ.synchronize()
    # 2D Surface
    gdim = 2
    # Groups together surface entities
    gmsh.model.addPhysicalGroup(gdim, [surface], 1, name='domain')
    # Groups together boundary entities
    gmsh.model.addPhysicalGroup(gdim-1, [surface], 1, name='edge')
    # Sets the mesh sizing
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", params['minsize'])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", params['maxsize'])
    # Generates mesh
    gmsh.model.mesh.generate(gdim)
    # Stuff for plotting the mesh

    # Converts gmsh model to a fenicsx mesh
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh_triplet = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=2)

    return mesh_triplet