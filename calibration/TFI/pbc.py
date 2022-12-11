from tkinter import E
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIModel
from tenpy.algorithms import dmrg

def energies_DMRG(L, g, n):
    #Initialize parameters
    model_params = dict(L=L, J=1, g=g, bc_MPS="finite", bc_x="periodic", order="folded", conserve=None)
    M = TFIModel(model_params)

    #Define initial state
    product_state = ["up"] * M.lat.N_sites
    psi0 = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    #Define DMRG parameters
    dmrg_params = {
        'mixer': False,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-10,
        'trunc_params': {
            'chi_max': 30,
            'svd_min': 1.e-10
        },
        'combine': True
    }

    energies = []
    states = []

    solution = dmrg.run(psi0, M, dmrg_params)
    E = solution['E']

    energies.append(E)
    states.append(psi0)

    for i in range(n):
        psi = psi0.copy()
        solution = dmrg.run(psi, M, dmrg_params, orthogonal_to=states)
    
        E = solution['E']
        energies.append(E)
        states.append(psi)

    return energies

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as arp
import warnings

def energies_exact(L, g, n):
    J = 1
    if L > 20:
        warnings.warn("Large L: Exact diagonalization might take a long time!")
    # get single site operaors
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sz_list = []
    for i_site in range(L):
        x_ops = [id] * L
        z_ops = [id] * L
        x_ops[i_site] = sx
        z_ops[i_site] = sz
        X = x_ops[0]
        Z = z_ops[0]
        for j in range(1, L):
            X = sparse.kron(X, x_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_xx = sparse.csr_matrix((2**L, 2**L))
    H_z = sparse.csr_matrix((2**L, 2**L))
    for i in range(L):
        H_xx = H_xx + sx_list[i] * sx_list[(i + 1) % L]
    for i in range(L):
        H_z = H_z + sz_list[i]
    H = -J * H_xx - g * H_z
    energies, _ = arp.eigsh(H, k=n+1, which='SA', return_eigenvectors=True, ncv=n+2)
    
    return energies

from math import pi, cos, sqrt

def energies_analytical(L, g, levels): #ASK DIRK
    #Calabrese et al
    
    arr = np.arange(-L/2, L/2)
    kn_arr = 2 * pi * (arr + 0.5) / L
    pn_arr = 2 * pi * arr / L
    ek_arr =  2 * np.sqrt(1 + g ** 2 - 2 * g * np.cos(kn_arr))
    ep_arr = 2 * np.sqrt(1 + g ** 2 - 2 * g * np.cos(pn_arr))

    ek_pairs = []
    ep_triplets = []
    for i in range(L):
        for j in range(i+1, L): #j>i
            ek_pairs.append(ek_arr[i] + ek_arr[j])
            for k in range(j+1, L): #k>j
                ep_triplets.append(ep_arr[i] + ep_arr[j] + ep_arr[k])
    ek_pairs = sorted(ek_pairs)
    ep_triplets = sorted(ep_triplets)

    energies = []
    for level in range(levels + 1):
        if level % 2 == 0: #Even sector
            if level == 0:
                eNS = -np.sum(ek_arr)/2
                energies.append(eNS)
            else:
                energies.append(eNS + ek_pairs[int(level/2)-1]) #0,1,2,3...

        elif level % 2 == 1: #Odd sector
            if level == 1:
                eR = - np.sum(ep_arr[arr != 0])/2 - (1 - g)
                energies.append(eR)
            else:
                energies.append(eR + ep_triplets[int((level-1)/2)]) #0,1,2,3...

    return energies

L = 10; g = 0.6; n = 10

# E_DMRG = energies_DMRG(L, g, n)
# print("DMRG: {}".format(E_DMRG))

E_exact = energies_exact(L, g, n)
E_exact_even = E_exact[::2]
E_exact_odd = E_exact[1::2]


E_analytical = energies_analytical(L, g, n)
E_analytical_even = E_analytical[::2]
E_analytical_odd = E_analytical[1::2]

# print("Exact: {}".format(E_exact))
# print("Analytical: {}".format(E_analytical))

print("Exact even: {}".format(E_exact_even))
print("Analytical even: {}".format(E_analytical_even))

# print("Exact odd: {}".format(E_exact_odd))
# print("Analytical odd: {}".format(E_analytical_odd))