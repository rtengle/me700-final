from mpi4py import MPI
from dolfinx.io import XDMFFile, gmshio
from dolfinx import fem
import gmsh
import pyvista
import numpy as np

# Initialize the mesh
gmsh.initialize()
gamma0 = 9*np.pi/180
print(gamma0)

# Add in a 2D disk
surface = gmsh.model.occ.addDisk(0, 0, 0, gamma0, gamma0)
gmsh.model.occ.synchronize()
gdim = 2
gmsh.model.addPhysicalGroup(gdim, [surface], 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(gdim)

gmsh_model_rank = 0
mesh_comm = MPI.COMM_WORLD
domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, gdim=gdim)

V = fem.functionspace(domain, ("Lagrange", 1))

from dolfinx.plot import vtk_mesh
import pyvista
pyvista.start_xvfb()

# Extract topology from mesh and create pyvista mesh
topology, cell_types, x = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, x)