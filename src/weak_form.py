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

params = dict()
params['gamma0'] = 9 * np.pi/180
params['minsize'] = 0.01
params['maxsize'] = 0.01
params['dt'] = 0.01

# Function that creates the gmsh model & mesh
model = create_gmsh(params)

# Converts gmsh model to a fenicsx mesh
gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(model, mesh_comm, gmsh_model_rank, gdim=2)
# Normal vector for weak formulation
n = ufl.FacetNormal(domain)

# Mixed Element of H and eta, doing quadratic elements because curvature is super important
He = element("Lagrange", domain.basix_cell(), 2, dtype=default_real_type)
Se = mixed_element([He, He])
S = fem.functionspace(domain, Se)
facets = fem.locate_dofs_topological(S, 1, facet_markers.indices)

# Define the boundary conditions on H
bcH = fem.dirichletbc(default_scalar_type(0), facets, S.sub(0))

# Define the boundary condition on eta
bceta = fem.dirichletbc(default_scalar_type(1), facets, S.sub(1))

bc = [bcH, bceta]

# Test functions
q, v = ufl.TestFunctions(S)

# Define our current and previous surface
u = fem.Function(S)
u0 = fem.Function(S)

# Split into H, eta and H0, eta0
H, eta = ufl.split(u)
H0, eta0 = ufl.split(u0)

# Initial conditions
u.x.array[:] = 0.0
u.sub(0).x.array[:] = 1.0 - 0.5
u.x.scatter_forward()

# Define our weak form

dt = params['dt']

FH = (
    ufl.inner(H - H0, q) * ufl.dx
    - dt/3 * ( ufl.inner( H**3, ufl.inner(ufl.grad(q) , ufl.grad(eta)) ) ) * ufl.dx
    + dt * ufl.inner( q, (H**3 * ufl.inner(ufl.grad(eta), n))) * ufl.ds
)
Feta = (
    (ufl.inner(eta,v) + ufl.dot(ufl.grad(v), ufl.grad(H))) * ufl.dx
    -  ufl.inner(v * ufl.grad(H), n)* ufl.ds
)
F = FH + Feta

# Set up and configure our non-linear problem solver

problem = NonlinearProblem(F, u, bcs=bc)

solver = NewtonSolver(MPI.COMM_WORLD, problem)
# Define the conergence criteria
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

# Modify the solver used, this is copied from the non-linear Poisson tutorial
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
sys = PETSc.Sys()  # type: ignore
# For factorisation prefer MUMPS, then superlu_dist, then default
use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
if sys.hasExternalPackage("mumps") and not use_superlu:
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "superlu_dist"
ksp.setFromOptions()

u0.x.array[:] = u.x.array

r = solver.solve(u)

pass