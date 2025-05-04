from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
from dolfinx.plot import vtk_mesh
from dolfinx import fem, default_scalar_type, default_real_type, mesh, plot
from basix.ufl import element, mixed_element
import gmsh
import pyvista
import numpy as np
import ufl
from petsc4py import PETSc

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from meshing import *

def create_solver(params, mesh_triplet, submesh_tuple):
    """This function returns the nonlinear solver used for the problem along with the variables and function spaces.
    Breakdown is as follows:
        S - 2D mixed element space with two components: H and eta. This function space represents the fluid surface
        T - 3D mixed element space with two components: theta and zeta. This function space represents the fluid temperature
    Our system is defined on a 3D mesh and a 2D submesh.
    """

    # Unpack the mesh and submesh variables
    omega, cell_markers, facet_markers = mesh_triplet
    gamma, entity_maps = submesh_tuple

    # Integration measure variables
    dx = ufl.Measure("dx", domain=omega)
    ds = ufl.Measure("ds", domain=omega, subdomain_data=facet_markers, subdomain_id=2)

    return solver, (s, s0, S), (z, z0, T)

def get_solver():
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
    return solver

def get_2D_elements(domain):
    # Mixed Element of H and eta, doing quadratic elements because curvature is super important
    He = element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type)
    Se = mixed_element([He, He])
    S = fem.functionspace(domain, Se)

    qa, qh = fem.TestFunctions(S)

    # Define our current and previous surface
    s = fem.Function(S)
    s0 = fem.Function(S)

    return (s, s0, S), (qa, qh)

def get_3D_elements(domain):
    T = fem.functionspace(domain, ("Laplace", 1))

    qt, qd = fem.TestFunctions(T)

    theta = fem.Function(T)
    theta0 = fem.Function(T)

    return (theta, theta0, T), (qt, qd)

def old_create_solver(params, triplet_2D, triplet_3D):
    # Get mesh information
    domain, cell_markers, facet_markers = mesh_triplet
    # Normal vector for weak formulation
    n = ufl.FacetNormal(domain)

    # Mixed Element of H and eta, doing quadratic elements because curvature is super important
    He = element("Lagrange", domain.basix_cell(), 1, dtype=default_real_type)
    Se = mixed_element([He, He])
    S = fem.functionspace(domain, Se)
    Z = fem.functionspace(domain, He)

    # Define the boundary conditions on H dofs
    Hfacets = fem.locate_dofs_topological(S.sub(0), 1, facet_markers.indices)
    bcH = fem.dirichletbc(default_real_type(params['Hpin']), Hfacets, S.sub(0))

    # Define the boundary condition on eta dofs
    etafacets = fem.locate_dofs_topological(S.sub(1), 1, facet_markers.indices)
    bceta = fem.dirichletbc(default_real_type(params['etapin']), etafacets, S.sub(1))

    bc = [bcH, bceta]

    # Test functions
    q, v = ufl.TestFunctions(S)

    # Theta
    x = ufl.SpatialCoordinate(domain)
    theta = theta_dist(x)

    # Define our current and previous surface
    u = fem.Function(S)
    u0 = fem.Function(S)

    # Split into H, eta and H0, eta0
    H, eta = ufl.split(u)
    H0, eta0 = ufl.split(u0)

    # Initial conditions
    u.x.array[:] = 0.0
    u.sub(0).x.array[:] = params['H0']
    u.x.scatter_forward()

    # Define our weak form

    dt = params['dt']

    FH = (
        ufl.inner(H - H0, q) * ufl.dx
        - dt * params['S']/3 * ( ufl.inner( H**3, ufl.inner(ufl.grad(q) , ufl.grad(eta)) ) ) * ufl.dx
        + dt * 1/2 * H**2 * ufl.inner(ufl.grad(q) , ufl.grad(theta)) * ufl.dx
        + dt * q * ufl.inner(H**3 * ufl.grad(eta) - H**2 * ufl.grad(theta), n) * ufl.ds
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

    return solver, u, u0, S