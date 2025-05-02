import pyvista as pv
import gmsh
import math
import sys

gmsh.initialize()
gmsh.clear()

# Copied from t1.py...
lc = 1e-2
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(0.1, 0, 0, lc, 2)
gmsh.model.geo.addPoint(0.1, 0.3, 0, lc, 3)
gmsh.model.geo.addPoint(0, 0.3, 0, lc, 4)
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(3, 2, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
ps = gmsh.model.addPhysicalGroup(2, [1])  # this is just the first area

# create second area
gmsh.model.geo.addPoint(0.3, 0, 0, lc, 5)
gmsh.model.geo.addPoint(0.4, 0, 0, lc, 6)
gmsh.model.geo.addPoint(0.4, 0.3, 0, lc, 7)
gmsh.model.geo.addPoint(0.3, 0.3, 0, lc, 8)
gmsh.model.geo.addLine(5, 6, 5)
gmsh.model.geo.addLine(7, 6, 6)
gmsh.model.geo.addLine(7, 8, 7)
gmsh.model.geo.addLine(8, 5, 8)
gmsh.model.geo.addCurveLoop([8, 5, -6, 7], 2)
gmsh.model.geo.addPlaneSurface([2], 2)
ps = gmsh.model.addPhysicalGroup(2, [2])  # this is just the first area

# must sync geo before meshing
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3)

# write the gmsh to disk
filename = "t1.msh"
gmsh.write(filename)

# read in the mesh using pyvista. plot it for fun just to show the physical maps
mesh = pv.read(filename)
mesh.plot(scalars="gmsh:physical", categories=True, show_edges=True, cmap="blues")
