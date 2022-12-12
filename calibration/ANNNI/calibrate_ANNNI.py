
import numpy as np
import pickle
import sys
import logging

sys.path.append("/home/juanjo/Code/dmrg")

from models.ANNNI import ANNNI

from tools.Processor.ProcessorANNNI import ProcessorANNNI
from tools import tools

from algorithms import DMRG
from algorithms import MERA

import logging

numba_logger = logging.getLogger('numba')

def test_DMRG(L, D, U, E, verbose=False):
    """
    Computes energies of ANNNI model using DMRG.
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
    model_params = {
        'conserve': 'Q',
        'bc_MPS': 'finite',
        'L': L,
        'D': D,
        'U': U,
        'E': E
    }
    sector_params = {
        'sectors': [0],
        'n_states_sector': 1, 
        'n_states': 1
    }

    simulation_path = 'calibration/ANNNI/'
    name = 'L{}_D{}_U{}_E{}'.format(float(L), float(D), float(U), float(E))

    try:
        #print("Trying to load {}".format(name))
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model = ANNNI(model_params)
        point, _ = DMRG.run(dmrg_params, model, sector_params, q=2, correlation_operators=("sx", "sx"))
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energies = round(point["energies"][0], 4)
    Processor = ProcessorANNNI(H_params={'D': [1], 'L': [L], 'E': [E], 'U': [U]}, sector_params=None, simulation_path=simulation_path)
    Processor.build_array()
    charges = Processor.compute_central_charges_fit()
    charges = round(charges[0][0][0][0], 4)

    return energies, charges
    

def test_MERA(D, U ,E):
    """
    Computes energies of ANNNP model using MERA.
    """
    mera_params = {
        'd': 2, # Site dimension
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

    simulation_path = 'calibration/ANNNI/'
    name = 'MERA_ANNNI_D{}_U{}_E{}.data'.format(float(D), float(U), float(E))
    try:
        #print("Trying to load {}".format(name))
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model_params = {'J': 1, 'D': D, 'U': U, 'E': E}
        model = tools.build_MERA_from_Model(ANNNI, model_params, 2)
        point, _ = MERA.run(mera_params, model)
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energy = round(point["energies"][-1], 4)
    charge = point["OPE_coefficients"][1,2,1].real

    return energy, charge


#####################################################

L = 6
D = 1
U = 1
E = 1

energiesDMRG, chargesDMRG = test_DMRG(L, D, U, E, verbose=True)
print("DMRG | Energy: {} | Energy density: {} | Central charge: {} ".format(energiesDMRG, energiesDMRG / L, chargesDMRG))

energyMERA, chargeMERA = test_MERA(D, U, E)
print("MERA | Energy: {} | Central charge: {}".format(energyMERA, chargeMERA))