"""
Simulation of frustration-free effects for the ANNNI model.
"""

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from models.ANNNI import ANNNI as Model
from algorithms import DMRG as Algorithm

from tools import tools
from tools.Processor.ProcessorANNNI import ProcessorANNNI
from tools.Plotter.PlotterANNNI import PlotterANNNI


# Model parameters 
model_params = {
    'conserve': 'Q',
    'bc_MPS': 'finite'
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

# Hamiltonian parameters
CUT_DIRECTION = 'vertical'
if CUT_DIRECTION == 'vertical':
    H_params = {
        'L': np.array([5, 6, 7]), 
        'D': tools.compute_fss_range(central_value=(1 + np.sqrt(3)), n=8, delta=0.02),
        'U': np.array([1.0]),
        'E': np.array([0.0])
    }
elif CUT_DIRECTION == 'horizontal':
    H_params = {
        'L': np.array([5, 6, 7]), 
        'D': np.array([1 + np.sqrt(3)]),
        'U': tools.compute_fss_range(central_value=0.97, n=8, delta=0.02),
        'E': np.array([0.0])
    }

# Sector parameters
sector_params = {
    'sectors': [0, 1], 
    'n_states_sector': 1, 
    'n_states': 2
}


def run(workers, simulation_path, parallel, use_cluster):
    """Run a pool workers in parallel"""

    worker = tools.build_worker(Model, Algorithm, model_params, algo_params, sector_params, simulation_path)
    iterable = np.stack(np.meshgrid(H_params['L'], H_params['D'], H_params['U'], H_params['E']), -1).reshape(-1,4)

    if parallel == True:
        pool = Pool(workers)
        pool.map(worker, iterable)
        
    elif parallel == False:
        for val in iterable:
            worker(val)
    

    """Process the data"""
    Processor = ProcessorANNNI(H_params, sector_params, simulation_path)
    charges = Processor.compute_central_charges()
    gaps = Processor.compute_gaps()
    correlations = Processor.compute_correlations()

    """Plot frustration-free effects"""
    if use_cluster == False:
        # Set critical point on which to compute scaling
        crit_x = 1 + np.sqrt(3)

        #Plot figures 
        Plotter = PlotterANNNI(H_params, simulation_path)

        # Vertical cut (change F):
        if CUT_DIRECTION == 'vertical':
            fixed_values = {'name_1': 'U', 'value_1': 1.0, 'name_2': 'E', 'value_2': 0.0}
            Plotter.plot_central_charges(charges, fixed_values, name_suffix='@' + CUT_DIRECTION)
            Plotter.plot_finite_size_gaps(gaps, fixed_values, crit_x=None, name_suffix='@' + CUT_DIRECTION)
        elif CUT_DIRECTION == 'horizontal':
            fixed_values = {'name_1': 'D', 'value_1': 1 + np.sqrt(3), 'name_2': 'E', 'value_2': 0.0}
            Plotter.plot_central_charges(charges, fixed_values, name_suffix='@' + CUT_DIRECTION)
            Plotter.plot_finite_size_gaps(gaps, fixed_values, crit_x=None, name_suffix='@' + CUT_DIRECTION)
