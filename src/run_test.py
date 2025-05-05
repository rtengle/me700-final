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
from solver_loop import *

def run_test(params):
    mesh_triplet = create_mesh(params)
    solver, function_triplet = create_solver(params, mesh_triplet)
    solver_loop(params, mesh_triplet, solver, function_triplet)