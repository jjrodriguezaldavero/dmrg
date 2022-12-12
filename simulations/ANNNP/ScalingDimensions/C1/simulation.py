"""
Simulation of the scaling relations of the ANNNP C1 point using MERA.
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
    'E_tol': 5e-8,

    'max_rounds': 5,
    'chi_init': 16,
    'chi_step': 2,
    'iters_init': 2000,
    'iters_step': -200,
    'layers_init': 1,
    'layers_step': 1,

    'scnum': 10
}

# Hamiltonian parameters
H_params = {
    'F': np.array([1.0]),
    'U': np.array([0.0]),
    'V': np.array([0.0])
}

def run(workers, simulation_path, parallel, use_cluster):
    """Run a pool workers in parallel"""

    use_checkpoint=True
    worker = tools.build_worker_MERA(Model, algo_params, simulation_path, use_checkpoint and use_cluster)
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
        
        array = Processor.build_MERA_array()

        Plotter = PlotterANNNP(H_params, simulation_path)

        scalings = array['scaling_dimensions'][0][0][0]
        exact_scalings = [0, 2/15, 2/15, 4/5, 2/15+1, 2/15+1, 2/15+1, 2/15+1, 4/3, 4/3]
        Plotter.plot_scalings(scalings, exact_scalings)