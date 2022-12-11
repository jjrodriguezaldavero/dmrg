"""
Simulation of the rough topography for the interpolating model.
"""

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from models.Sutherland import Sutherland as Model
from algorithms import DMRG as Algorithm

from tools import tools
from tools.Processor.ProcessorSutherland import ProcessorSutherland
from tools.Plotter.PlotterSutherland import PlotterSutherland


# Model parameters 
model_params = {
    'conserve': 'Q',
    'bc_MPS': 'finite',
    'bc_x': 'periodic',
    'order': 'folded'
}

# Algorithm parameters
algo_params = {
    'mixer': True,
    'max_E_err': 1.e-5,
    'trunc_params': {
        'svd_min': 1.e-10,
        'chi_max': 500,
    },
    'max_sweeps': 50,
    'max_hours': 1
}

# Hamiltonian parameters for the point C1: (U, F) = (0, 1) performing a cut along the F parameter
H_params = {
    'L': np.array([30, 36]), 
    'theta': np.linspace(start=0, stop=2*np.pi, num=64)[0:16]
}

# Sector parameters
sector_params = {
    'sectors': [0, 1, 2], 
    'n_states_sector': 1, 
    'n_states': 1
}


def run(workers, simulation_path, parallel, use_cluster):
    """Run a pool workers in parallel"""

    worker = tools.build_worker(Model, Algorithm, model_params, algo_params, sector_params, simulation_path)
    iterable = np.stack(np.meshgrid(H_params['L'], H_params['theta']), -1).reshape(-1,2)

    if parallel == True:
        pool = Pool(workers)
        pool.map(worker, iterable)
        
    elif parallel == False:
        for val in iterable:
            worker(val)
    

    """Process the data"""
    Processor = ProcessorSutherland(H_params, sector_params, simulation_path)
    charges = Processor.compute_central_charges()

    """Plot rough topography"""
    if use_cluster == False:
        Plotter = PlotterSutherland(H_params, simulation_path)
        Plotter.plot_central_charges(charges, fixed_values=None)
        Plotter.plot_central_charges_polar(charges, H_params, charge_type='normal')
