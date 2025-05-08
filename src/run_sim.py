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

def run_sim(params):
    """Main method used to simulate fluid surface deformation

    Arguments
    ---------
    params : dict
        Dictionary of simulation parameters. See README for info about all the parameter
    """
    mesh_triplet = create_mesh(params)
    solver, function_triplet = create_solver(params, mesh_triplet)
    final_tuple = solver_loop(params, mesh_triplet, solver, function_triplet)

def check_steady_state():
    """Test function used to check if steady-state analytic solution is matched
    """
    S = 0.1
    FK = 2

    def H(x):
        r2 = x[0]**2 + x[1]**2
        return 1+r2-r2**2

    def theta(x):
        r2 = x[0]**2 + x[1]**2
        return -64/3*S*(1/2*r2 + 1/4*r2**2 - 1/6*r2**3) + FK*(1+r2-r2**2)

    params = dict()
    params['flat'] = True
    params['gamma0'] = 1
    params['minsize'] = 4e-2
    params['maxsize'] = 4e-2
    params['Hpin'] = lambda x: default_real_type(1)
    params['etapin'] = lambda x: default_real_type(-12)
    params['H0'] = H
    params['dt'] = 5e-3
    params['S'] = S
    params['N'] = 20
    params['theta0'] = theta
    params['F/K'] = FK
    params['degree'] = 4
    params['rtol'] = 1e-12
    params['filename'] = 'data'
    params['foldername'] = 'results'
    params['figurename'] = 'H_animation'
    params['plot'] = True

    mesh_triplet = create_mesh(params)
    solver, function_triplet = create_solver(params, mesh_triplet)
    s, s0, S = solver_loop(params, mesh_triplet, solver, function_triplet)

    s0.sub(0).interpolate(H)
    SH, dofs = S.sub(0).collapse()
    emax = (s.x.array[dofs] - s0.x.array[dofs]).max()
    L2 = np.linalg.norm(s.x.array[dofs] - s0.x.array[dofs])

    print(emax)
    print(L2)

    return emax, L2

