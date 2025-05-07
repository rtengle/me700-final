from run_sim import run_sim
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
params['minsize'] = 5e-2
params['maxsize'] = 5e-2
params['Hpin'] = lambda x: default_real_type(1)
params['etapin'] = lambda x: default_real_type(-12)
params['H0'] = H
params['dt'] = 1e-3
params['S'] = S
params['N'] = 20
params['theta0'] = theta
params['F/K'] = FK
params['degree'] = 2
params['rtol'] = 1e-6
params['filename'] = 'data'
params['foldername'] = 'results'
params['figurename'] = 'H_animation'
params['plot'] = True

run_sim(params)