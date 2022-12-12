"""
Multi-Scale Entanglement Renormalization Ansatz (MERA).
"""

import numpy as np

from tools.MERAEngine import doVarMERA, doConformalMERA


def run(mera_params, model):
    """
    Runs the MERA.
    """

    """
    1. Initialize tensors
    """
    d = mera_params['d']
    chi = mera_params['chi_init']
    mera_params['layers_init'] = int(max(mera_params['layers_init'], np.ceil(np.log(chi) / (2 * np.log(4)))))
    hbig = model
    hamAB = [0] * (mera_params['layers_init'] + 2)
    hamBA = [0] * (mera_params['layers_init'] + 2)
    D = d**2
    hamAB[0] = (hbig.transpose(0, 1, 3, 2, 4, 5, 7, 6)).reshape(D, D, D, D)
    hamBA[0] = (hbig.transpose(1, 0, 2, 3, 5, 4, 6, 7)).reshape(D, D, D, D)

    totLv = mera_params['layers_init'] + 1
    chiZ = np.zeros(totLv + 1, dtype=int)
    chiZ[0] = hamAB[0].shape[0]
    chimidZ = np.zeros(totLv + 1, dtype=int)
    chimidZ[0] = hamAB[0].shape[0]
    for k in range(totLv):
        chiZ[k + 1] = min(chi, chiZ[k] * chimidZ[k])
        chimidZ[k + 1] = min(chi - 2, chiZ[k + 1])

    wC = [0] * (mera_params['layers_init'] + 1)
    vC = [0] * (mera_params['layers_init'] + 1)
    uC = [0] * (mera_params['layers_init'] + 1)
    for k in range(totLv):
        wC[k] = np.random.rand(chiZ[k], chimidZ[k], chiZ[k + 1])
        vC[k] = np.random.rand(chiZ[k], chimidZ[k], chiZ[k + 1])
        uC[k] = (np.eye(chiZ[k]**2, chimidZ[k]**2)).reshape(chiZ[k], chiZ[k], chimidZ[k], chimidZ[k])

    rhoAB = [0] * (mera_params['layers_init'] + 2)
    rhoBA = [0] * (mera_params['layers_init'] + 2)
    rhoAB[0] = np.eye(chiZ[0]**2).reshape(chiZ[0], chiZ[0], chiZ[0], chiZ[0])
    rhoBA[0] = np.eye(chiZ[0]**2).reshape(chiZ[0], chiZ[0], chiZ[0], chiZ[0])
    for k in range(totLv):
        rhoAB[k + 1] = np.eye(chiZ[k + 1]**2).reshape(chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1])
        rhoBA[k + 1] = np.eye(chiZ[k + 1]**2).reshape(chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1])
        hamAB[k + 1] = np.zeros((chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]), dtype=np.complex128)
        hamBA[k + 1] = np.zeros((chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]), dtype=np.complex128)

    tensors = {'hamAB': hamAB, 'hamBA': hamBA, 'rhoAB': rhoAB, 'rhoBA': rhoBA, 'wC': wC, 'vC': vC, 'uC': uC}


    """
    2. Perform variational optimization rounds
    """
    # Set parameters
    E_tol = mera_params['E_tol']
    iters = mera_params['iters_init']
    layers = mera_params['layers_init']
    sciter = 4

    for _ in range(mera_params['max_rounds']):
        # Perform variational optimization
        energies, tensors, converged = doVarMERA(tensors, E_tol, iters, chi, chi - 2, layers, sciter)
        if converged == True: break

        # Update parameters
        chi += mera_params['chi_step']
        iters += mera_params['iters_step']
        layers += mera_params['layers_step']

    # Compute the conformal data from the optimized MERA
    scnum = mera_params['scnum']
    scDims, scOps, Cfusion = doConformalMERA(wC[-1], uC[-1], vC[-1], rhoBA[-1], scnum)

    point = {
        'energies': energies, 
        'scaling_dimensions': scDims, 
        'scaling_operators': scOps, 
        'OPE_coefficients': Cfusion
    }

    return point, tensors