import numpy as np
import pickle
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/home/juanjo/Code/dmrg")

from models.Sutherland import Sutherland

from tools import tools
from tools.Processor.ProcessorSutherland import ProcessorSutherland

from algorithms import DMRG
from algorithms import MERA


def test_DMRG(L, theta, verbose=False):
    """
    Computes energies of Sutherland model using DMRG.
    """

    if verbose: logging.basicConfig(level=logging.INFO)

    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-5,
        'trunc_params': {
            'svd_min': 1.e-10,
            'chi_max': 50,
        }
    }
    model_params = {'J': 1, 'theta': theta,'L': L,
            'conserve': None,
            'bc_MPS': 'finite'}
    
    sector_params = {
        'sectors': [0],
        'n_states_sector': 1, 
        'n_states': 1
    }

    simulation_path = 'calibration/Sutherland/'
    name = 'L{}_theta{}'.format(float(L), float(theta))

    try:
        #print("Trying to load {}".format(name))
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model = Sutherland(model_params)
        point, _ = DMRG.run(dmrg_params, model, sector_params)
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energies = round(point["energies"][0], 4)
    Processor = ProcessorSutherland(H_params={'L': [L], 'theta':[theta]}, sector_params=None, simulation_path=simulation_path)
    Processor.build_array()
    charges = Processor.compute_central_charges_fit()
    charges = round(charges[0][0], 4)

    return energies, charges
    

def test_MERA(theta, plot=True):
    """
    Computes energies of Sutherland model using MERA.
    """

    mera_params = {
        'd': 3, # Local Hilbert space dimension
        'E_tol': 5e-3, # Convergence criteria

        # Round parameters:
        'max_rounds': 5, 
        'chi_init': 8,
        'chi_step': 2,
        'iters_init': 2400,
        'iters_step': -200,
        'layers_init': 1,
        'layers_step': 1,

        'scnum': 10 # Number of scaling dimensions to save
    }

    simulation_path = 'calibration/Sutherland/'
    name = 'MERA_theta{}'.format(round(float(theta), 4))
    try:
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model_params = {'theta': theta}
        model = tools.build_MERA_from_Model(Sutherland, model_params, 3)
        point, _ = MERA.run(mera_params, model)
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energy = round(point["energies"][-1], 4)
    charge = point["OPE_coefficients"][1,2,1].real
    scalings = point["scaling_dimensions"]

    if plot == True:
        sns.set_style('whitegrid')
        plt.plot(range(len(scalings)), scalings, 'rx', label="Numerical")
        plt.title('Sutherland scaling dimensions')
        plt.xlabel('k')
        plt.ylabel(r'Scaling Dims: $\Delta_k$')
        plt.legend()
        plt.show()

    return energy, charge


#####################################################

L = 20
theta = np.pi/4

energiesDMRG, chargesDMRG = test_DMRG(L, theta, verbose=False)
print("DMRG | Energy: {} | Energy density: {} | Central charge: {} ".format(energiesDMRG, energiesDMRG / L, chargesDMRG))

energyMERA, chargeMERA = test_MERA(theta)
print("MERA | Energy: {} | Central charge: {}".format(energyMERA, chargeMERA))