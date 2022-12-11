"""
Simulation of the scaling relations of the Quantum Torus chain point using MERA.
"""

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from models.Sutherland import Sutherland as Model
from algorithms import MERA as Algorithm

from tools import tools
from tools.Processor.ProcessorSutherland import ProcessorSutherland
from tools.Plotter.PlotterSutherland import PlotterSutherland


# Model parameters 
model_params = {}

# Algorithm parameters
algo_params = {
    'E_tol': 1e-7,
    'max_rounds': 5,
    'chi_init': 16,
    'chi_step': 2,
    'iters_init': 3000,
    'iters_step': -200,
    'layers_init': 1,
    'layers_step': 1,
    'sciter': 4, # iterations of power method to find density matrix,
    'scnum': 20,
    'continue': False,
    'verbose': True
}

# Hamiltonian parameters
H_params = {
    'L': np.array([2]),
    'theta': np.linspace(start=0, stop=np.pi/4, num=9)
}

# Sector parameters (does not apply)
sector_params = None

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
    scalings = Processor.build_scalings()["scalings"]

    """Plot Scaling dimensions"""
    if use_cluster == False:

        #Plot figures 
        Plotter = PlotterSutherland(H_params, simulation_path)
        Plotter.plot_scalings(scalings)