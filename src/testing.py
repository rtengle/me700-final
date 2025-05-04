from meshing import *
from weak_form import *

params = dict()
params['gamma0'] = 9 * np.pi/180
params['2Dminsize'] = 1e-2
params['2Dmaxsize'] = 1e-2
params['3Dminsize'] = 1e-2
params['3Dmaxsize'] = 1e-2
params['Hpin'] = 1
params['etapin'] = 0
params['H0'] = 1
params['dt'] = 1e-4
params['S'] = 1e-1
params['N'] = 50

mesh_tuple = create_mesh(params)
submesh_tuple = create_submesh(mesh_tuple)

# results = create_solver(params, mesh_tuple, submesh_tuple)