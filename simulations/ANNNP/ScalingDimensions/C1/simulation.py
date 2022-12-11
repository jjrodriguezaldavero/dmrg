"""
Simulation of the scaling relations of the ANNNP C1 point using MERA.
"""

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from models.ANNNP import ANNNP as Model
from algorithms import MERA as Algorithm

from tools import tools
from tools.Processor.ProcessorANNNP import ProcessorANNNP
from tools.Plotter.PlotterANNNP import PlotterANNNP


# Model parameters 
model_params = {}

# Algorithm parameters
algo_params = {
    'd': 3,
    'E_tol': 5e-7,
    'max_rounds': 5,
    'chi_init': 16,
    'chi_step': 2,
    'iters_init': 2000,
    'iters_step': -200,
    'trlayers_init': 1,
    'trlayers_step': 1,
    'sciter': 4, # iterations of power method to find density matrix,
    'scnum': 20,
    'continue': False,
    'verbose': True
}

# Hamiltonian parameters
H_params = {
    'L': np.array([2.0]),
    'F': np.array([1.0]),
    'U': np.array([0.0]),
    'V': np.array([0.0])
}

# Sector parameters (does not apply)
sector_params = {}

def run(workers, simulation_path, parallel, use_cluster):
    """Run a pool workers in parallel"""

    worker = tools.build_worker(Model, Algorithm, model_params, algo_params, sector_params, simulation_path)
    iterable = np.stack(np.meshgrid(H_params['L'], H_params['F'], H_params['U'], H_params['V']), -1).reshape(-1,4)

    if parallel == True:
        pool = Pool(workers)
        pool.map(worker, iterable)
        
    elif parallel == False:
        for val in iterable:
            worker(val)
    
    """Plot Scaling dimensions"""
    if use_cluster == False:

        Processor = ProcessorANNNP(H_params, sector_params, simulation_path)
        scalings = Processor.build_scalings()["scalings"]
        
        #Plot figures 
        Plotter = PlotterANNNP(H_params, simulation_path)
        Plotter.plot_scalings(scalings)