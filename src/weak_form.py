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

def get_functionspace(params, domain):
    """Internal function to get the mixed element function space for H and eta
    """
    He = element("Lagrange", domain.basix_cell(), params['degree'], dtype=default_real_type)
    Se = mixed_element([He, He])
    S = fem.functionspace(domain, Se)
    return S

def get_bc(params, S, domain, facet_markers):
    """Internal function used to get the dirichlet BC for H and eta
    """
    x = ufl.SpatialCoordinate(domain)
    # Define the boundary conditions on H dofs
    Hfacets = fem.locate_dofs_topological(S.sub(0), 1, facet_markers.find(1))
    bcH = fem.dirichletbc(params['Hpin'](x), Hfacets, S.sub(0))

    # Define the boundary condition on eta dofs
    etafacets = fem.locate_dofs_topological(S.sub(1), 1, facet_markers.find(1))
    bceta = fem.dirichletbc(params['etapin'](x), etafacets, S.sub(1))

    return [bcH, bceta]

def create_solver(params, mesh_triplet):
    """Creates the solver, function space, and functions used to simulate the fluid surface.

    Parameters
    ----------
    params : dict
        Dictionary containing all user-defined settings for the simulation. Needs to contain at least the following:
        theta0 : Callable
            Function that returns a UFL expression for the temperature given a UFL spatial coordinate 
        F/K : float
            Value for the constant F/K determining the radial temperature gradient
        dt : float
            Time interval between steps
        S : float
            Surface tension coefficient
        H0 : float
            Initial surface height
        Hpin : float
            Fixed height at boundary
        etapin : float
            Fixed laplacian of surface height at the boundaries
        degree : int
            Degree of mesh element
        rtol : float
            Relative tolerance of solver

    Returns
    -------
    solver
        Solver object that iteratively solves for the correct film surface
    function_triplet : tuple
        Tuple containing the following:
        s
            Mixed function for the current fluid surface
        s0
            Mixed function for the previous fluid surface
        S
            Mixed function space for the fluid surface
    
    """
    # Get mesh information
    domain, cell_markers, facet_markers = mesh_triplet

    S = get_functionspace(params, domain)

    # Define our current and previous surface
    s = fem.Function(S)
    s0 = fem.Function(S)

    # Split into H, eta and H0, eta0
    H, eta = ufl.split(s)
    H0, eta0 = ufl.split(s0)

    # Test functions
    q, v = ufl.TestFunctions(S)

    s.sub(0).name = 'Height'
    s.sub(1).name = 'Curvature'

    # Define our weak form

    dt = params['dt']
    Sp = params['S']

    # Normal vector for weak formulation
    x = ufl.SpatialCoordinate(domain)
    n = ufl.FacetNormal(domain)

    # Surface temperature function
    theta = params['theta0'](x) - H * params['F/K']

    # Weak formulation for surface height
    FH = (
        ufl.inner(H - H0, q) * ufl.dx
        - dt * Sp/3 * H**3 * ufl.inner(ufl.grad(q), ufl.grad(eta)) * ufl.dx
        + dt * 1/2 * H**2 * ufl.inner(ufl.grad(q), ufl.grad(theta)) * ufl.dx
        + dt * q * ufl.inner(H**3 * ufl.grad(eta) - H**2 * ufl.grad(theta), n) * ufl.ds
    )
    # Weak formulation for eta-H relationship
    Feta = (
        (ufl.inner(eta,v) + ufl.dot(ufl.grad(v), ufl.grad(H))) * ufl.dx
        -  ufl.inner(v * ufl.grad(H), n) * ufl.ds
    )
    # Combine together to get complete weak formulation
    F = FH + Feta

    # Get the boundary conditions for the system
    bc = get_bc(params, S, domain, facet_markers)

    # Initial conditions
    s.x.array[:] = 0.0
    if type(params['H0'](x)) is np.float64:
        s.sub(0).x.array[:] = params['H0'](x)
    else:
        s.sub(0).interpolate(params['H0'])
    s.x.scatter_forward()

    # Set up and configure our non-linear problem solver

    problem = NonlinearProblem(F, s, bcs=bc)
    solver = get_solver(params, problem)

    return solver, (s, s0, S)

def get_solver(params, problem):
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    # Define the conergence criteria
    solver.convergence_criterion = "incremental"
    solver.rtol = params['rtol']
    solver.report = True

    # Modify the solver used, this is copied from the Cahn-Hilliard tutorial
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