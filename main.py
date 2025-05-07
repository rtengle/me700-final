from run_sim import *

params = dict()
params['gamma0'] = 1
params['minsize'] = 5e-2
params['maxsize'] = 5e-2
params['Hpin'] = 1
params['etapin'] = 0
params['H0'] = 1
params['dt'] = 1e-1
params['S'] = 1e-1
params['N'] = 10
params['theta0'] = lambda x: ufl.exp(-1e-2*x[0]**2 - 1e-2*x[1]**2)
params['degree'] = 2
params['F/K'] = 2
params['rtol'] = 1e-6
params['filename'] = 'data'
params['foldername'] = 'results'
params['figurename'] = 'H_animation'

run_sim(params)