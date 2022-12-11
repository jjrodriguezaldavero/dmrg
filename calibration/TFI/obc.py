from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.algorithms import dmrg

def energies_DMRG(L, g, n):
    #Initialize parameters
    model_params = dict(L=L, J=1, g=g, bc_MPS="finite", bc_x="open", order="default", conserve=None)
    M = TFIChain(model_params)

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
    for i in range(L - 1):
        H_xx = H_xx + sx_list[i] * sx_list[(i + 1) % L]
    for i in range(L):
        H_z = H_z + sz_list[i]
    H = -J * H_xx - g * H_z
    energies, _ = arp.eigsh(H, k=n+1, which='SA', return_eigenvectors=True, ncv=20)
    
    return energies

L = 10; g = 0.6; n = 1

E_DMRG = energies_DMRG(L, g, n)
print("DMRG: {} | Gap: {}".format(E_DMRG, E_DMRG[1] - E_DMRG[0]))

E_exact = energies_exact(L, g, n)
print("Exact: {} | Gap: {}".format(E_exact, E_exact[1] - E_exact[0]))
