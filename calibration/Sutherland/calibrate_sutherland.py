import numpy as np
import pickle
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/home/juanjo/Code/dmrg")

from models.Sutherland import Sutherland
from tools.Processor.ProcessorSutherland import ProcessorSutherland

from tenpy.algorithms.exact_diag import ExactDiag

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
            'bc_MPS': 'finite'}#,
            # 'bc_x': 'periodic',
            # 'order': 'folded'}
    
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
    

def test_MERA(theta, iters=[2000, 1800, 1600], chis=[6, 8, 12], chimids=[4, 6, 8], plot=False, checkpoint_name=None):
    """
    Computes energies of Sutherland model using MERA.
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

    simulation_path = 'calibration/Sutherland/'
    name = 'MERA_it{}_theta{}'.format(sum(iters), float(theta))
    try:
        #print("Trying to load {}".format(name))
        with open(simulation_path + 'data/' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model_params = {'theta': theta, 'conserve': None, 'bc_MPS': 'finite', 'bc_x': 'periodic'}
        model = Sutherland(model_params)

        d = 3
        diag = ExactDiag(model)
        diag.build_full_H_from_mpo()
        H_two = diag.full_H.to_ndarray()
        H_four = (np.kron(np.eye(d), np.kron(H_two, np.eye(d))) + (1/2) * (np.kron(np.eye(d**2), H_two) + np.kron(H_two, np.eye(d**2)))).reshape(d,d,d,d,d,d,d,d)

        if checkpoint_name is not None:
            with open(simulation_path + 'data/' + checkpoint_name, 'rb') as f:
                checkpoint = pickle.load(f)
            point, _ = MERA.run(mera_params, H_four, checkpoints=True, path=simulation_path + 'data/' + name, checkpoint=checkpoint)
        else:
            point, _ = MERA.run(mera_params, H_four, checkpoints=True, path=simulation_path + 'data/' + name)

        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energies = round(point["energies"].real, 4)
    charges = [round(point["Cess"][0].real, 4), round(point["Cess"][1].real, 4), round(point["Cess"][2].real, 4)]
    scalings = point["scDims"]

    if plot == True:
        sns.set_style('whitegrid')
        plt.plot(range(len(scalings)), scalings, 'rx', label="Numerical")
        plt.title('Sutherland scaling dimensions')
        plt.xlabel('k')
        plt.ylabel(r'Scaling Dims: $\Delta_k$')
        plt.legend()
        plt.show()

    return energies, charges


#####################################################

L = 20
theta = np.pi/4

# energiesDMRG, chargesDMRG = test_DMRG(L, F, U, V, verbose=True)
# print("DMRG | Energy: {} | Energy density: {} | Central charge: {} ".format(energiesDMRG, energiesDMRG / L, chargesDMRG))

# energiesMERA, chargesMERA = test_MERA(
#     F, U, V, chis=[14, 16, 18], chimids=[12, 14, 16], iters=[2000,1800,1600], 
#     plot=True, checkpoint_name="MERA_it5400_F1.0_U0.0_V0.0@checkpoint_2")

energiesMERA, chargesMERA = test_MERA(
    theta, chis=[18], chimids=[14,16,18], iters=[2000,2000,2000], 
    plot=True)

print("MERA | Energy: {} | Fusion coefficients: {}, {}, {}".format(energiesMERA, chargesMERA[0], chargesMERA[1], chargesMERA[2]))