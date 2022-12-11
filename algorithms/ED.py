"""
Exact Diagonalization algorithm.
"""

from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.networks.mps import MPS


def run(model, n_states):
    """
    Runs ED.
    """
    product_state = ["0"] * model.lat.N_sites
    psi0 = MPS.from_product_state(model.lat.mps_sites(), product_state, bc=model.lat.bc_MPS)
    charge_sector = psi0.get_total_charge(True)

    ED = ExactDiag(model, charge_sector=charge_sector, max_size=2.e8)
    ED.build_full_H_from_mpo()
    ED.full_diagonalization() 
    eigenvalues = ED.E
    energies = eigenvalues[0:n_states]

    point = {"energies": energies, "ED_data": ED}

    return point