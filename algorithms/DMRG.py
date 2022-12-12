"""
Density Matrix Renormalization Group algorithm.
"""

import numpy as np

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS


def random_initial_product_state(q, L, sector):
    """
    Defines an initial random product state within a given sector and with a charge q.
    """
    state = list(np.random.randint(q, size = L-1))
    charge = sum(state)%q
    state.append((sector-charge)%q)
    charge = sum(state)%q
    assert charge==sector
    return state

def run(dmrg_params, model, sector_params, q, correlation_operators):
    """
    Runs DMRG.
    """
    assert sector_params['n_states'] <= len(sector_params['sectors']) * sector_params['n_states_sector']
    
    L = model.H_MPO.L

    states = []
    energies = []
    for sector in sector_params['sectors']:
        initial_state = random_initial_product_state(q, L, sector)
        psi0 = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc=model.lat.bc_MPS)

        states_sector = []
        energies_sector = []
        for state in range(sector_params['n_states_sector']):
            psi = psi0.copy()
            print('Computing state {} of sector {}...'.format(state, sector))
            results = dmrg.run(psi, model, dmrg_params, orthogonal_to=states_sector)
            states_sector.append(psi)
            energies_sector.append(results['E'])

        states.append(states_sector)
        energies.append(energies_sector)

    flatten_list = lambda elements: [y for x in elements for y in x]
    states = flatten_list(states)
    energies = flatten_list(energies)

    order_of_states = np.argsort(energies)
    states = [states[i] for i in order_of_states]

    energies = [energies[i] for i in order_of_states][:sector_params['n_states']]
    entropies = states[0].entanglement_entropy()
    correlations = states[0].correlation_function(correlation_operators[0], correlation_operators[1])
    canonical_flags = [states[i].canonical_flag for i in order_of_states]

    point = {
        "energies": energies, 
        "entropy": entropies, 
        "correlation": correlations, 
        "convergences": canonical_flags
    }

    return point, states