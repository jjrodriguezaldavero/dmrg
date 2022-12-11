import numpy as np
import pickle
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/home/juanjo/Code/dmrg")

from models.ANNNP import ANNNP
from tools.Processor.ProcessorANNNP import ProcessorANNNP

from tenpy.algorithms.exact_diag import ExactDiag

from algorithms import DMRG
from algorithms import MERA


def test_DMRG(L, F, U, V, verbose=False):
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
    model_params = {'J': 1, 'F': F, 'U': U, 'V': V,'L': L,
            'conserve': None,
            'bc_MPS': 'finite'}#,
            # 'bc_x': 'periodic',
            # 'order': 'folded'}
    
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
        point, _ = DMRG.run(dmrg_params, model, sector_params)
        with open(simulation_path + 'data/' + name, 'wb+') as f:
            pickle.dump(point, f, 2)

    energies = round(point["energies"][0], 4)
    Processor = ProcessorANNNP(H_params={'J': 1, 'L': [L], 'F': [F], 'U': [U], 'V': [V]}, sector_params=None, simulation_path=simulation_path)
    Processor.build_array()
    charges = Processor.compute_central_charges_fit()
    charges = round(charges[0][0][0][0], 4)

    return energies, charges
    

def test_MERA(mera_params, model):
    """
    Computes energies of ANNNP model using MERA.
    """
    mera_params = {
    'd': 3,
    'E_tol': 1e-7,
    'max_rounds': 5,
    'chi_init': 16,
    'chi_step': 2,
    'iters_init': 3000,
    'iters_step': -200,
    'trlayers_init': 1,
    'trlayers_step': 1,
    'sciter': 4, # iterations of power method to find density matrix,
    'scnum': 20,
    'continue': False,
    'verbose': True
}

    simulation_path = 'calibration/ANNNP/'
    name = 'MERA_it{}_F{}_U{}_V{}'.format(sum(iters), float(F), float(U), float(V))
    try:
        #print("Trying to load {}".format(name))
        with open(simulation_path + 'data/a' + name, 'rb') as f:
            point = pickle.load(f)
    except:
        model_params = {'J': 1, 'F': F, 'U': U, 'V': V,
            'conserve': None,
            'bc_MPS': 'finite',
            'bc_x': 'periodic'}
            #'order': 'folded'}
        model = ANNNP(model_params)#.build_MERA_hamiltonian()

        ###
        diag = ExactDiag(model)
        diag.build_full_H_from_mpo()
        H_two = diag.full_H.to_ndarray()
        # print(H_two)
        d=3
        H_four = (np.kron(np.eye(d), np.kron(H_two, np.eye(d))) + (1/2) * (np.kron(np.eye(d**2), H_two) + np.kron(H_two, np.eye(d**2)))).reshape(d,d,d,d,d,d,d,d)
        ###
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
    exactScalingsC1 = [0, 2/15, 2/15, 4/5, 2/15+1, 2/15+1, 2/15+1, 2/15+1, 4/3, 4/3]
    exactScalingsC2 = [2*scale for scale in exactScalingsC1]

    if plot == True:
        sns.set_style('whitegrid')
        plt.plot(range(len(scalings)), scalings, 'rx', label="Numerical")
        #plt.plot(range(scalings), exactScalingsC2, 'o', fillstyle='none', label="Exact")
        plt.title('ANNNP scaling dimensions')
        plt.xlabel('k')
        plt.ylabel(r'Scaling Dims: $\Delta_k$')
        plt.legend()
        plt.show()

    return energies, charges


#####################################################

L = 20
F = 0
U = -1
V = 0

# energiesDMRG, chargesDMRG = test_DMRG(L, F, U, V, verbose=True)
# print("DMRG | Energy: {} | Energy density: {} | Central charge: {} ".format(energiesDMRG, energiesDMRG / L, chargesDMRG))

# energiesMERA, chargesMERA = test_MERA(
#     F, U, V, chis=[14, 16, 18], chimids=[12, 14, 16], iters=[2000,1800,1600], 
#     plot=True, checkpoint_name="MERA_it5400_F1.0_U0.0_V0.0@checkpoint_2")





energiesMERA, chargesMERA = test_MERA(mera_params, model)

print("MERA | Energy: {} | Fusion coefficients: {}, {}, {}".format(energiesMERA, chargesMERA[0], chargesMERA[1], chargesMERA[2]))