"""
Definition of the Chiral Axial Next-Nearest-Neighbor Potts Hamiltonian.
"""

import numpy as np

from tenpy.models.model import CouplingMPOModel
from models.networks.ANNNPSite import ANNNPSite


class ChiralANNNP(CouplingMPOModel):
    def init_sites(self, model_params):
        "Import site definition"
        conserve = model_params.get("conserve", None)
        site = ANNNPSite(conserve)
        return site

    def init_terms(self, model_params):
        "Define hamiltonian"
        J = model_params.get("J", 1)
        F = model_params.get("F", 1)
        U = model_params.get("U", 1)
        V = model_params.get("V", 1)
        phiJ = model_params.get("phiJ", 1)
        phiF = model_params.get("phiF", 1)
        phiU = model_params.get("phiU", 1)
        phiV = model_params.get("phiV", 1)

        for u in range(len(self.lat.unit_cell)):
            # Single terms
            self.add_onsite(-F * np.exp(-1.0j * phiF), u, "t", plus_hc=True)

        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            # Nearest neighbors terms
            self.add_coupling(-J * np.exp(-1.0j * phiJ), u1, "s", u2, "sD", dx, plus_hc=True)
            self.add_coupling(U * np.exp(-1.0j * phiU), u1, "t", u2, "t", dx, plus_hc=True)
            self.add_coupling(V * np.exp(-1.0j * phiV), u1, "t", u2, "tD", dx, plus_hc=True)