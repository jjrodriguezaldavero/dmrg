"""
Definition of the Axial Next-Nearest-Neighbor Ising Hamiltonian.
"""

from tenpy.models.model import CouplingMPOModel
from models.networks.ANNNISite import ANNNISite
import numpy as np

class ANNNI(CouplingMPOModel):
    def init_sites(self, model_params):
        "Import site definition"
        conserve = model_params.get("conserve", None)
        site = ANNNISite(conserve)
        return site

    def init_terms(self, model_params):
        "Define hamiltonian"
        D = model_params.get("D", 1)
        U = model_params.get("U", 1)
        E = model_params.get("E", 1)

        for u in range(len(self.lat.unit_cell)):
            # Single terms
            self.add_onsite(D, u, "sz", plus_hc=False)

        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            # Nearest neighbors terms
            self.add_coupling(U, u1, "sz", u2, "sz", dx, plus_hc=False)
            self.add_coupling(E, u1, "sx", u2, "sx", dx, plus_hc=False)