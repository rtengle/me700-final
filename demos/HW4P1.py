import ufl
import numpy
import matplotlib as mpl

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, nls, log, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

import pyvista

# This is a time-stepping non-linear Poisson Equation taking the form:
#
# dH/dt - S*div( H^3 grad(H) ) = 0
# H = 0 @ Boundary
#
# This is iterated step-by-step using an implicit scheme via a Nonlinear solver

# Define temporal parameters
t = 0  # Start time
T = 1.0  # Final time
num_steps = 70 # Number of steps
dt = T / num_steps  # time step size
S = 1 # Growth factor or coefficient

# This is the initial condition for our system. It obeys the boundary conditions set above.
def initial_condition(x, a=5):
    return numpy.sin(numpy.pi*x[0])*numpy.sin(numpy.pi*x[1])

# Create a unit square with 10 elements along each axis for 100 elements total
domain = mesh.create_unit_square(MPI.COMM_WORLD, 50, 50)

# Create a 1D lagrange function space
V = fem.functionspace(domain, ("Lagrange", 1))

# Get the boundary dimension from the domain mesh
fdim = domain.topology.dim - 1
# Locate the boundaries of the mesh
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool))
# Prescribe a Dirichlet BC of H = 0 at the boundary
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Define the function being solved for H
H = fem.Function(V)
H.name = "H"
H.interpolate(initial_condition)

# Define the function describing the previous timestep's H
H0 = fem.Function(V)
H0.interpolate(initial_condition)

# Define the test function v
v = ufl.TestFunction(V)

# Define the functional F that relates H and v
F = (H-H0)*v*ufl.dx + S * dt * H**3 * ufl.dot(ufl.grad(H), ufl.grad(v)) * ufl.dx

# Load the problem and boundary conditions to a non-linear problem class
problem = NonlinearProblem(F, H, bcs=[bc])

# Create a non-linear solver for the problem defined
solver = NewtonSolver(MPI.COMM_WORLD, problem)
# Define the conergence criteria
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

# Modify the solver used, this is copied from the non-linear Poisson tutorial
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
# A bunch of options I don't understand
opts[f"{option_prefix}ksp_type"] = "gmres"
opts[f"{option_prefix}ksp_rtol"] = 1.0e-8
opts[f"{option_prefix}pc_type"] = "hypre"
opts[f"{option_prefix}pc_hypre_type"] = "boomeramg"
opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"
ksp.setFromOptions()

# Start the pyvista frame buffer
pyvista.start_xvfb()

# Create an unstructured grid for the plot
grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

# Create a plotter that will create a .gif
plotter = pyvista.Plotter()
plotter.open_gif("Hred_time.gif", fps=10)

# Stores the H data
grid.point_data["H"] = H.x.array
# I think warp adds a height map on a grid
warped = grid.warp_by_scalar("H", factor=1)

# Set the color map
viridis = mpl.colormaps.get_cmap("viridis").resampled(25)

# This is the arguments for the color bar
sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
             position_x=0.1, position_y=0.8, width=0.8, height=0.1)

# Adds in the mesh with a color bar and height
renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                            cmap=viridis, scalar_bar_args=sargs,
                            clim=[0, max(H.x.array)])

for i in range(num_steps):
    try:
        # Increases time step
        t += dt
        # Solves for H using the previous time step H0
        r = solver.solve(H)
        # Prints which iterating it's currently on
        print(f"Step {int(t / dt)}: num iterations: {r[0]}")
        # Updates the H0 to the sovled time step
        H0.x.array[:] = H.x.array

        # Replaces the height map from before
        new_warped = grid.warp_by_scalar("H", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["H"][:] = H.x.array
        # Writes the plot to the next frame
        plotter.write_frame()
    except:
        # Closes the plot if an error is encountered so I can see what failed
        plotter.close()
        quit()

# Closes the plot
plotter.close()
