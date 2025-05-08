from meshing import *
from weak_form import *
from solver_loop import *
from run_sim import *
import pytest
import numpy as np
import ufl

params = dict()
params['flat'] = False
params['gamma0'] = np.pi/4
params['minsize'] = 5e-2
params['maxsize'] = 5e-2
params['Hpin'] = lambda x: default_real_type(1)
params['etapin'] = lambda x: default_real_type(0)
params['H0'] = lambda x: default_real_type(1)
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
    """Check if meshing process defines an outer boundary
    """
    domain, cell_markers, facet_markers = create_mesh(params)
    x = facet_markers.find(1)
    assert True

def test_functionspace():
    """Checks to see if get_functionspace runs without errors
    """
    domain, cell_markers, facet_markers = create_mesh(params)
    S = get_functionspace(params, domain)
    assert True

def test_bc():
    """Checks to see if get_bc produces the proper bc values
    """
    mesh_triplet = create_mesh(params)
    # Get mesh information
    domain, cell_markers, facet_markers = mesh_triplet

    S = get_functionspace(params, domain)

    bcH, bceta = get_bc(params, S, domain, facet_markers)
    assert bcH.g.value == 1
    assert bceta.g.value == 0

def test_weak_form():
    """Checks if weak_form runs without errors
    """
    mesh_triplet = create_mesh(params)
    solver, function_triplet = create_solver(params, mesh_triplet)
    assert True

def test_run_sim():
    """Checks if run_sim runs without errors
    """
    run_sim(params)
    assert True

def test_steady_state():
    """Checks if SS solution stays SS
    """
    emax, L2 = check_steady_state()
    assert emax < 1e-3
    assert L2 < 0.5

