
import numpy as np
import pickle
import sys
import logging

sys.path.append("/home/juanjo/Code/dmrg")

from models.ANNNI import ANNNI
from models.ANNNI import ANNNIMERA

from tools.Processor.ProcessorANNNI import ProcessorANNNI

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
        'F': D,
        'U': U,
        'V': E
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
        point, _ = DMRG.run(dmrg_params, model, sector_params)
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energies = round(point["energies"][0], 4)
    Processor = ProcessorANNNI(H_params={'D': 1, 'L': [L], 'E': [E], 'U': [U]}, sector_params=None, simulation_path=simulation_path)
    Processor.build_array()
    charges = Processor.compute_central_charges_fit()
    charges = round(charges[0][0][0][0], 4)

    return energies, charges
    

def test_MERA(D, U ,E, iters=[2000, 1800, 1600], chis=[6, 8, 12], chimids=[4, 6, 8]):
    """
    Computes energies of ANNNP model using MERA.
    """
    mera_params = {
        'numiter': 0,  # number of variatonal iterations
        'refsym': True,  # impose reflection symmetry
        'numtrans': 1,  # number of transitional layers
        'dispon': True,  # display convergence data
        'sciter': 4,  # iterations of power method to find density matrix,
        'iters': iters,
        'chis': chis,
        'chimids': chimids,
    }

    simulation_path = 'calibration/ANNNI/'
    name = 'MERA_it{}_F{}_U{}_V{}'.format(sum(iters), float(D), float(U), float(E))
    try:
        #print("Trying to load {}".format(name))
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model_params = {'J': 1, 'D': D, 'U': U, 'E': E}
        model = ANNNI(model_params).build_MERA_hamiltonian(model_params)
        point = MERA.run(mera_params, model, d=2)
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energies = round(point["energies"].real, 4)
    charges = [round(point["Cess"][0].real, 4), round(point["Cess"][1].real, 4), round(point["Cess"][2].real, 4)]
    scalings = point["scDims"]

    return energies, charges


#####################################################

L = 50
D = 1
U = 1
E = 1

# energiesDMRG, chargesDMRG = test_DMRG(L, D, U, E, verbose=True)
# print("DMRG | Energy: {} | Energy density: {} | Central charge: {} ".format(energiesDMRG, energiesDMRG / L, chargesDMRG))

energiesMERA, chargesMERA = test_MERA(D, U, E, chis=[6, 8, 10], chimids=[4, 6, 8], iters=[2000,1800,1600])
print("MERA | Energy: {} | Fusion coefficients: {}, {}, {}".format(energiesMERA, chargesMERA[0], chargesMERA[1], chargesMERA[2]))