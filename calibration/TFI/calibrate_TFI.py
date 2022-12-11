import numpy as np
import pickle
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/home/juanjo/Code/dmrg")

from tenpy.models.tf_ising import TFIModel
from tenpy.algorithms.exact_diag import ExactDiag
from tools.Processor.OLD.ProcessorTFI import ProcessorTFI


from algorithms import DMRG
from algorithms import MERA


def test_DMRG(L, J, g, verbose=False):
    """
    Computes energies of ANNNP model using DMRG.
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
        'conserve': None,
        'bc_MPS': 'finite',
        'bc_x': 'periodic',
        'order': 'folded',
        'L': 4,
        'J': J,
        'g': g
    }
    sector_params = {
        'sectors': [0],
        'n_states_sector': 1, 
        'n_states': 1
    }

    simulation_path = 'calibration/TFI/'
    name = 'L{}_J{}_g{}'.format(float(L), float(J), float(g))

    try:
        #print("Trying to load {}".format(name))
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model = TFIModel(model_params)
        point, _ = DMRG.run(dmrg_params, model, sector_params, q=2, correlation_operators=("Sigmax", "Sigmax"))
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energies = round(point["energies"][0], 4)
    Processor = ProcessorTFI(H_params={'L': [L], 'J': [J], 'g': [g]}, sector_params=None, simulation_path=simulation_path)
    Processor.build_array()
    charges = Processor.compute_central_charges_fit()
    charges = round(charges[0][0][0], 4)

    return energies, charges
    

def test_MERA(g, iters=[2000, 1800, 1600], chis=[6, 8, 12], chimids=[4, 6, 8], plot=False):
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

    simulation_path = 'calibration/TFI/'
    name = 'MERA_it{}_g{}'.format(sum(iters), float(g))
    try:
        #print("Trying to load {}".format(name))
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model_params = {
            'conserve': None,
            'bc_MPS': 'finite',
            'bc_x': 'periodic',
            'order': 'folded',
            'L': 2,
            'J': J,
            'g': g
        }
        model = TFIModel(model_params)
        hamiltonian = model.build_MERA_hamiltonian()
        diag = ExactDiag(model)
        diag.build_full_H_from_mpo()
        H_two = diag.full_H.to_ndarray()/2
        print(H_two)
        d=2
        H_four = (np.kron(np.eye(d), np.kron(H_two, np.eye(d))) + (1/2) * (np.kron(np.eye(d**2), H_two) + np.kron(H_two, np.eye(d**2)))).reshape(2,2,2,2,2,2,2,2)
        point, _ = MERA.run(mera_params, H_four, checkpoints=True, path=simulation_path + 'data/' + name, d=2)
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energies = round(point["energies"].real, 4)
    charges = [round(point["Cess"][0].real, 4), round(point["Cess"][1].real, 4), round(point["Cess"][2].real, 4)]
    scalings = point["scDims"]

    if plot == True:
        sns.set_style('whitegrid')
        plt.plot(range(10), scalings, 'rx')
        plt.title('ANNNP scaling dimensions')
        plt.xlabel('k')
        plt.ylabel('Scaling Dims: Delta_k')
        plt.show()

    return energies, charges


#####################################################

L = 4
J = 1
g = 1

# energiesDMRG, chargesDMRG = test_DMRG(L, J, g, verbose=True)
# print("DMRG | Energy: {} | Energy density: {} | Central charge: {} ".format(energiesDMRG, energiesDMRG / L, chargesDMRG))

energiesMERA, chargesMERA = test_MERA(g, chis=[12, 14, 16], chimids=[6, 8, 10], iters=[200,200,200], plot=True)
print("MERA | Energy: {} | Fusion coefficients: {}, {}, {}".format(energiesMERA, chargesMERA[0], chargesMERA[1], chargesMERA[2]))