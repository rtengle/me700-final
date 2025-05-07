from meshing import *
from weak_form import *
from solver_loop import *
from run_sim import *
import pytest
import numpy as np
import ufl

params = dict()
params['gamma0'] = np.pi/4
params['minsize'] = 5e-2
params['maxsize'] = 5e-2
params['Hpin'] = 1
params['etapin'] = 0
params['H0'] = 1
params['dt'] = 1e-1
params['S'] = 1e-1
params['N'] = 33
params['theta0'] = lambda x: ufl.exp(-1e-2*x[0]**2 - 1e-2*x[1]**2)
params['degree'] = 2
params['F/K'] = 2
params['rtol'] = 1e-6
params['filename'] = 'data'
params['foldername'] = 'results'
params['figurename'] = 'H_animation'
params['plot'] = True

def test_meshing():
    domain, cell_markers, facet_markers = create_mesh(params)
    x = facet_markers.find(1)
    assert True

def test_functionspace():
    domain, cell_markers, facet_markers = create_mesh(params)
    S = get_functionspace(params, domain)
    assert True

def test_bc():
    mesh_triplet = create_mesh(params)
    # Get mesh information
    domain, cell_markers, facet_markers = mesh_triplet

    S = get_functionspace(params, domain)

    bcH, bceta = get_bc(params, S, facet_markers)
    assert bcH.g.value == 1
    assert bceta.g.value == 0

def test_weak_form():
    mesh_triplet = create_mesh(params)
    solver, function_triplet = create_solver(params, mesh_triplet)
    assert True

def test_run_sim():
    run_sim(params)
    assert True

