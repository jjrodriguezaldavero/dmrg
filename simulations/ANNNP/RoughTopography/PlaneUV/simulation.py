"""
Simulation of the rough topography for the upper F-V plane
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
    'max_sweeps': 50,
    'max_hours': 0.5
}

# Hamiltonian parameters for the plane UxV given by (-1.5, 1.5) x (-1.5, 1.5)
H_params = {
    'L': np.array([50]), 
    'F': np.array([0.0]),
    'U': np.linspace(start=-1.5, stop=1.5, num=61),
    'V': np.linspace(start=-1.5, stop=1.5, num=61),
}

# Sector parameters
sector_params = {
    'sectors': [0], 
    'n_states_sector': 1, 
    'n_states': 1
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
    entropies = Processor.compute_central_entropies()
    charges = Processor.compute_central_charges_fit()
    
    """Plot rough topography"""
    if use_cluster == False:
        Plotter = PlotterANNNP(H_params, simulation_path)
        fixed_values = {'name_1': 'L', 'value_1': 50, 'name_2': 'F', 'value_2': 0.0}
        Plotter.plot_topography(entropies, fixed_values, name_suffix='@entropy_UV', cmap='coolwarm',
            title="Entanglement entropies of the UV plane")
        Plotter.plot_topography(charges, fixed_values, name_suffix='@charge_UV',
            title="Central charges of the UV plane")