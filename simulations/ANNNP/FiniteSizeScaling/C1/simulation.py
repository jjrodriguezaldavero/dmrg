"""
Simulation of Finite Size Scaling for the ANNNP model at the C1 point.
"""

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from models.ANNNP import ANNNP as Model
from algorithms import DMRG as Algorithm

from tools import tools
from tools.Processor.ProcessorANNNP import ProcessorANNNP
from tools.Plotter.PlotterANNNP import PlotterANNNP


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
    },
    'max_sweeps': 30,
    'max_hours': 0.5
}

# Hamiltonian parameters for the point C1: (U, F) = (0, 1) performing a cut along the F parameter
H_params = {
    'L': np.array([3, 64, 70, 76, 82, 88, 94, 100]), 
    'F': tools.compute_fss_range(central_value=1, n=8, delta=0.004),
    'U': np.array([0]),
    'V': np.array([0.0])
}

# Sector parameters
sector_params = {
    'sectors': [0, 1, 2], 
    'n_states_sector': 1, 
    'n_states': 2
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
    betas = Processor.compute_betas()
    correlations = Processor.compute_correlations()

    """Plot Finite Size Scaling"""
    if use_cluster == False:
        # Set critical point on which to compute scaling
        crit_x = 1

        #Plot figures 
        Plotter = PlotterANNNP(H_params, simulation_path)

        # Plot figures for V = 0.0
        fixed_values = {'name_1': 'U', 'value_1': 0, 'name_2': 'V', 'value_2': 0.0}
        name_suffix = '@V_00'
        Plotter.plot_central_charges(charges, fixed_values, name_suffix=name_suffix)
        Plotter.plot_finite_size_gaps(gaps, fixed_values, crit_x=crit_x, name_suffix=name_suffix)
        Plotter.plot_finite_size_betas(betas, fixed_values, crit_x=crit_x, name_suffix=name_suffix)
        Plotter.plot_finite_size_correlations(correlations, fixed_values, crit_x=crit_x, name_suffix=name_suffix)

        # # For V = 0.2
        # fixed_values = {'name_1': 'U', 'value_1': 0, 'name_2': 'V', 'value_2': 0.2}
        # name_suffix = '@V_02'
        # Plotter.plot_central_charges(charges, fixed_values, name_suffix=name_suffix)
        # Plotter.plot_finite_size_gaps(gaps, fixed_values, crit_x=crit_x, name_suffix=name_suffix)
        # Plotter.plot_finite_size_betas(betas, fixed_values, crit_x=crit_x, name_suffix=name_suffix)
        # Plotter.plot_finite_size_correlations(correlations, fixed_values, crit_x=crit_x, name_suffix=name_suffix)

        # # For V = -0.2
        # fixed_values = {'name_1': 'U', 'value_1': 0, 'name_2': 'V', 'value_2': -0.2}
        # name_suffix = '@V_-02'
        # Plotter.plot_central_charges(charges, fixed_values, name_suffix=name_suffix)
        # Plotter.plot_finite_size_gaps(gaps, fixed_values, crit_x=crit_x, name_suffix=name_suffix)
        # Plotter.plot_finite_size_betas(betas, fixed_values, crit_x=crit_x, name_suffix=name_suffix)
        # Plotter.plot_finite_size_correlations(correlations, fixed_values, crit_x=crit_x, name_suffix=name_suffix)
