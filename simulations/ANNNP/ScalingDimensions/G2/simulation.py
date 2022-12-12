"""
Simulation of the scaling relations of the ANNNP G1 point using MERA.
"""

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from models.ANNNP import ANNNP as Model

from tools import tools
from tools.Processor.ProcessorANNNP import ProcessorANNNP
from tools.Plotter.PlotterANNNP import PlotterANNNP


# Algorithm parameters
algo_params = {
    'd': 3,
    'E_tol': 5e-7,

    'max_rounds': 5,
    'chi_init': 16,
    'chi_step': 2,
    'iters_init': 2000,
    'iters_step': -200,
    'layers_init': 1,
    'layers_step': 1,

    'scnum': 20
}

# Hamiltonian parameters
H_params = {
    'F': np.array([1.0 + np.sqrt(3)]),
    'U': np.array([0.9675]),
    'V': np.array([0.0])
}

def run(workers, simulation_path, parallel, use_cluster):
    """Run a pool workers in parallel"""

    worker = tools.build_worker_MERA(Model, algo_params, simulation_path)
    iterable = np.stack(np.meshgrid(H_params['F'], H_params['U'], H_params['V']), -1).reshape(-1,3)

    if parallel == True:
        pool = Pool(workers)
        pool.map(worker, iterable)
        
    elif parallel == False:
        for val in iterable:
            worker(val)
    
    """Plot Scaling dimensions"""
    if use_cluster == False:
        sector_params = {}
        Processor = ProcessorANNNP(H_params, sector_params, simulation_path)
        
        #Work out a scaling processor and plots