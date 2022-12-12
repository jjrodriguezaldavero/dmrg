"""
Simulation of frustration-free effects for the ANNNP model at the G2 point.
"""

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from models.ANNNP import ANNNP as Model

from tools import tools
from tools.Processor.ProcessorANNNP import ProcessorANNNP
from tools.Plotter.PlotterANNNP import PlotterANNNP


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
    'max_E_err': 1.e-6,
    'trunc_params': {
        'svd_min': 1.e-10,
        'chi_max': 100,
    }
}

# Hamiltonian parameters for the point G1: (U, F) = (0, 0)
H_params = {
    #'L': np.array([4, 8, 12, 16, 24]), 
    'L': np.array([4, 8, 12, 16]),
    'F': tools.compute_fss_range(central_value=(1 + np.sqrt(3)), n=8, delta=0.2),
    'U': tools.compute_fss_range(central_value=(1), n=8, delta=0.2),
    'V': np.array([0.0])
}

# Sector parameters
sector_params = {
    'sectors': [0, 1, 2], 
    'n_states_sector': 2, 
    'n_states': 6
}


def run(workers, simulation_path, parallel, use_cluster):
    """Run a pool workers in parallel"""

    worker = tools.build_worker_DMRG(Model, model_params, algo_params, sector_params, simulation_path)
    iterable = np.stack(np.meshgrid(H_params['L'], H_params['F'], H_params['U'], H_params['V']), -1).reshape(-1,4)

    if parallel == True:
        pool = Pool(workers)
        pool.map(worker, iterable)
        
    elif parallel == False:
        for val in iterable:
            worker(val)
    

    """Process the data"""
    Processor = ProcessorANNNP(H_params, sector_params, simulation_path)
    charges = Processor.compute_central_charges()
    gaps = Processor.compute_gaps()

    """Plot frustration-free effects"""
    if use_cluster == False:

        #Plot figures 
        Plotter = PlotterANNNP(H_params, simulation_path)
        fixed_value = {'name': 'V', 'value': 0.0}
        Plotter.plot_frustration_free_fit_2D(gaps[0], fixed_value, type='logsurface', r=0.5*(np.sqrt(3)-1))

        # Work out a plotting function
