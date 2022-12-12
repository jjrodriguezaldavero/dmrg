import pickle
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/home/juanjo/Code/dmrg")

from models.ANNNP import ANNNP

from algorithms import DMRG
from algorithms import MERA

from tools.Processor.ProcessorANNNP import ProcessorANNNP
from tools import tools


def test_DMRG(L, F, U, V, verbose=False):
    """
    Computes energies of ANNNP model using DMRG.
    """

    if verbose: logging.basicConfig(level=logging.INFO)

    q = 3
    correlation_operators = ('s', 'sD')

    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-5,
        'trunc_params': {
            'svd_min': 1.e-10,
            'chi_max': 50,
        }
    }

    model_params = {
        'conserve': None,
        'bc_MPS': 'finite',
        'L': L,
        'F': F,
        'U': U,
        'V': V
    }
    
    H_params = {
            'L': [L], 
            'F': [F], 
            'U': [U], 
            'V': [V]}
    

    sector_params = {
        'sectors': [0],
        'n_states_sector': 1, 
        'n_states': 1
    }

    simulation_path = 'calibration/ANNNP/'
    name = 'L{}_F{}_U{}_V{}'.format(float(L), float(F), float(U), float(V))

    try:
        #print("Trying to load {}".format(name))
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model = ANNNP(model_params)
        point, _ = DMRG.run(dmrg_params, model, sector_params, q, correlation_operators)
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energies = round(point["energies"][0], 4)
    Processor = ProcessorANNNP(H_params, sector_params=None, simulation_path=simulation_path)
    Processor.build_array()
    charges = Processor.compute_central_charges_fit()
    charges = round(charges[0][0][0][0], 4)

    return energies, charges
    

def test_MERA(F, U, V, plot=True):
    """
    Computes energies of ANNNP model using MERA.
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

    simulation_path = 'calibration/ANNNP/'
    name = 'MERA_ANNNP_F{}_U{}_V{}.data'.format(float(F), float(U), float(V))
    try:
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model_params = {'J': 1, 'F': F, 'U': U, 'V': V}
        model = tools.build_MERA_from_Model(ANNNP, model_params, 3)
        point, _ = MERA.run(mera_params, model)
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energy = round(point["energies"][-1], 4)
    charge = point["OPE_coefficients"][1,2,1].real
    scalings = point["scaling_dimensions"]
    exact_scalings = [0, 2/15, 2/15, 4/5, 2/15+1, 2/15+1, 2/15+1, 2/15+1, 4/3, 4/3]

    if plot == True:
        sns.set_style('whitegrid')
        plt.plot(range(len(scalings)), scalings, 'rx', label="Numerical")
        plt.plot(range(len(scalings)), exact_scalings, 'o', fillstyle='none', label="Exact")
        plt.title('ANNNP scaling dimensions')
        plt.xlabel('k')
        plt.ylabel(r'Scaling Dims: $\Delta_k$')
        plt.show()

    return energy, charge


#####################################################

L = 7
F = 1
U = 0
V = 0

energiesDMRG, chargesDMRG = test_DMRG(L, F, U, V, verbose=False)
print("DMRG | Energy: {} | Energy density: {} | Central charge: {} ".format(energiesDMRG, energiesDMRG / L, chargesDMRG))

energyMERA, chargeMERA = test_MERA(F, U, V)
print("MERA | Energy: {} | Central charge: {}".format(energyMERA, chargeMERA))