"""
Simulation of the phase diagram for the interpolating model.
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
    'bc_MPS': 'finite'
}

# Algorithm parameters
algo_params = {
    'mixer': True,
    'max_E_err': 1.e-6,
    'trunc_params': {
        'svd_min': 1.e-10,
        'chi_max': 500,
    },
    'max_sweeps': 50,
    'max_hours': 1
}

# Hamiltonian parameters for the point C1: (U, F) = (0, 1) performing a cut along the F parameter
H_params = {
    'L': np.array([12, 18, 48, 60, 72, 84, 96, 120]), 
    #'theta': np.array([0.0, (1/4)*np.pi, (1/2)*np.pi, (3/4)*np.pi, np.pi, (5/4)*np.pi])
    'theta': np.linspace(start=0, stop=np.pi/4, num=13)
}

# Sector parameters
sector_params = {
    'sectors': [0, 1, 2], 
    'n_states_sector': 2, 
    'n_states': 3
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
    charges_fit = Processor.compute_central_charges_fit()
    gaps = Processor.compute_gaps()
    correlations = Processor.compute_correlations()

    """Plot phase diagram"""
    if use_cluster == False:
        # Set critical point on which to compute scaling
        crit_x = None

        plot_params = {
            'xlabel'
        }

        #Plot figures 
        fixed_values = None
        Plotter = PlotterSutherland(H_params, simulation_path)
        Plotter.plot_central_charges(charges, fixed_values)
        Plotter.plot_central_charges(charges_fit, fixed_values, name_suffix="@fit")
        Plotter.plot_finite_size_gaps(gaps, fixed_values, crit_x)
        Plotter.plot_finite_size_correlations(correlations, fixed_values, crit_x)
        Plotter.plot_fitted_gaps(gaps)
        Plotter.plot_gap_scalings(gaps, H_params, fixed_values)
        Plotter.plot_correlation_scalings(correlations, H_params, fixed_values)
        
        
