from run_sim import run_sim, check_steady_state
import ufl
import numpy as np
from dolfinx import default_real_type

# This is a very quick tutorial on how to use the codebase. Below is a set of parameters describing the following experiment:
#
# The film is on a circular shell with a range of ± pi/4 rad. 
# 
# We have S = 0.1 and F/k = 2. We apply a flat gaussian pulse projected onto the bottom surface with theta0
# 
# The sides are pinned at H = 1 and del^2 H = eta = 0 with a starting height of 1
#
# We have a time step of dt = 0.1 and take a total N = 33 steps.
#
# We use degree 2 elements with a minimum and maximum element size of 0.05
#
# Finally we save our results to the results folder under the name data, set plot to true, and save our figure as H_animation

params = dict()
params['flat'] = False
params['gamma0'] = 1
params['minsize'] = 5e-2
params['maxsize'] = 5e-2
params['Hpin'] = lambda x: default_real_type(1)
params['etapin'] = lambda x: default_real_type(0)
params['H0'] = lambda x: default_real_type(1)
params['dt'] = 5e-3
params['S'] = 5e-2
params['N'] = 50
params['theta0'] = lambda x: 2*ufl.exp(-100*( (x[0]/params['gamma0'])**2 + (x[1]/params['gamma0'])**2 ))
params['degree'] = 2
params['F/K'] = 2
params['rtol'] = 1e-6
params['filename'] = 'data'
params['foldername'] = 'results'
params['figurename'] = 'H_animation'
params['plot'] = True

run_sim(params)