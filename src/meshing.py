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

def create_mesh(params):
    gmsh.initialize()
    # Defines bottom surface
    gmsh.model.occ.addCircle(0, 0, 0, params['gamma0'], 1)
    gmsh.model.occ.addCurveLoop([1], 1)
    # Defines top surface
    gmsh.model.occ.addCircle(0, 0, 1,params['gamma0'], 2)
    gmsh.model.occ.addCurveLoop([2], 2)

    # Links the two together
    gmsh.model.occ.addThruSections([1, 2], 1)
    gmsh.model.occ.addSurfaceLoop([1], 3)
    gmsh.model.occ.synchronize()
    # Specifies where the boundaries are
    gmsh.model.addPhysicalGroup(3, [1], 1, name='domain')
    gmsh.model.addPhysicalGroup(2, [1], 1, name='bottom')
    gmsh.model.addPhysicalGroup(2, [2], 2, name='top')
    gmsh.model.addPhysicalGroup(2, [3], 3, name='sides')
    # Specifies where the edges are
    gmsh.model.addPhysicalGroup(1, [1], 1, name='bottom ring')
    gmsh.model.addPhysicalGroup(1, [2], 2, name='top ring')
    # Sets the mesh sizing
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", params['3Dminsize'])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", params['3Dmaxsize'])
    # Generates mesh
    gmsh.model.mesh.generate(3)

    # Converts gmsh model to a fenicsx mesh
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    mesh_tuple = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=3)

    return mesh_tuple

def create_submesh(mesh_tuple):
    # Unpacks mesh triplet
    omega, cell_markers, facet_markers, edge_markers = mesh_tuple

    # Our submesh will be the top surface, defined by a tag of 2
    gamma, g2o = mesh.create_submesh(omega, 2, facet_markers.find(2))

    # Create the mapping from omega to gamma
    facet_imap = omega.topology.index_map(facet_markers.dim)
    num_facets = facet_imap.size_local + facet_imap.num_ghosts
    o2g = np.full(num_facets, -1)
    o2g[g2o] = np.arange(len(g2o))
    entity_maps = {gamma: o2g}

    return gamma, entity_maps