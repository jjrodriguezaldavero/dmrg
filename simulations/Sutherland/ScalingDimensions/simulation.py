"""
Simulation of the scaling relations of the Quantum Torus chain point using MERA.
"""

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from models.Sutherland import Sutherland as Model

from tools import tools
from tools.Processor.ProcessorSutherland import ProcessorSutherland
from tools.Plotter.PlotterSutherland import PlotterSutherland


# Algorithm parameters
algo_params = {
    'E_tol': 5e-7, # Convergence criteria

    # Round parameters:
    'max_rounds': 5, 
    'chi_init': 16,
    'chi_step': 2,
    'iters_init': 2000,
    'iters_step': -200,
    'layers_init': 1,
    'layers_step': 1,

    'scnum': 20 # Number of scaling dimensions to save
}

# Hamiltonian parameters
H_params = {
    'theta': np.linspace(start=0, stop=np.pi/4, num=9)
}

def run(workers, simulation_path, parallel, use_cluster):
    """Run a pool workers in parallel"""

    worker = tools.build_worker_MERA(Model, algo_params, simulation_path)
    iterable = np.stack(np.meshgrid(H_params['theta']), -1).reshape(-1,1)

    if parallel == True:
        pool = Pool(workers)
        pool.map(worker, iterable)
        
    elif parallel == False:
        for val in iterable:
            worker(val)
    

    """Process the data"""
    sector_params = {}
    Processor = ProcessorSutherland(H_params, sector_params, simulation_path)
    
    # Work out a scalings processor and plots