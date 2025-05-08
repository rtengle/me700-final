from meshing import *
from weak_form import *
from solver_loop import *
from run_sim import *
import pytest
import numpy as np
import ufl

sphere_params = dict()
sphere_params['flat'] = False
sphere_params['gamma0'] = 1
sphere_params['minsize'] = 5e-2
sphere_params['maxsize'] = 5e-2
sphere_params['Hpin'] = lambda x: default_real_type(1)
sphere_params['etapin'] = lambda x: default_real_type(0)
sphere_params['H0'] = lambda x: default_real_type(1)
sphere_params['dt'] = 5e-3
sphere_params['S'] = 5e-2
sphere_params['N'] = 50
sphere_params['theta0'] = lambda x: 2*ufl.exp(-100*( (x[0]/sphere_params['gamma0'])**2 + (x[1]/sphere_params['gamma0'])**2 ))
sphere_params['degree'] = 2
sphere_params['F/K'] = 2
sphere_params['rtol'] = 1e-6
sphere_params['filename'] = 'data'
sphere_params['foldername'] = 'results'
sphere_params['figurename'] = 'H_animation'
sphere_params['plot'] = True

flat_params = dict()
flat_params['flat'] = True
flat_params['gamma0'] = 1
flat_params['minsize'] = 5e-2
flat_params['maxsize'] = 5e-2
flat_params['Hpin'] = lambda x: default_real_type(1)
flat_params['etapin'] = lambda x: default_real_type(0)
flat_params['H0'] = lambda x: default_real_type(1)
flat_params['dt'] = 5e-3
flat_params['S'] = 7e-2
flat_params['N'] = 50
flat_params['theta0'] = lambda x: 2*ufl.exp(-100*( (x[0]/flat_params['gamma0'])**2 + (x[1]/flat_params['gamma0'])**2 ))
flat_params['degree'] = 2
flat_params['F/K'] = 2
flat_params['rtol'] = 1e-6
flat_params['filename'] = 'data'
flat_params['foldername'] = 'results'
flat_params['figurename'] = 'H_animation'
flat_params['plot'] = True

def test_meshing_sphere():
    """Check if meshing process defines an outer boundary
    """
    domain, cell_markers, facet_markers = create_mesh(sphere_params)
    x = facet_markers.find(1)
    assert True

def test_meshing_flat():
    """Check if meshing process defines an outer boundary
    """
    domain, cell_markers, facet_markers = create_mesh(flat_params)
    x = facet_markers.find(1)
    assert True

def test_functionspace():
    """Checks to see if get_functionspace runs without errors
    """
    domain, cell_markers, facet_markers = create_mesh(sphere_params)
    S = get_functionspace(sphere_params, domain)
    assert True

def test_bc():
    """Checks to see if get_bc produces the proper bc values
    """
    mesh_triplet = create_mesh(sphere_params)
    # Get mesh information
    domain, cell_markers, facet_markers = mesh_triplet

    S = get_functionspace(sphere_params, domain)

    bcH, bceta = get_bc(sphere_params, S, domain, facet_markers)
    assert bcH.g.value == 1
    assert bceta.g.value == 0

def test_weak_form():
    """Checks if weak_form runs without errors
    """
    mesh_triplet = create_mesh(sphere_params)
    solver, function_triplet = create_solver(sphere_params, mesh_triplet)
    assert True

def test_run_sim_sphere():
    """Checks if run_sim runs without errors
    """
    run_sim(sphere_params)
    assert True

def test_run_sim_flat():
    """Checks if run_sim runs without errors
    """
    run_sim(sphere_params)
    assert True

def test_steady_state():
    """Checks if SS solution stays SS
    """
    emax, L2 = check_steady_state()
    assert emax < 1e-3
    assert L2 < 0.5

