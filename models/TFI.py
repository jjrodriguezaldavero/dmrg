"""
Definition of the Quantum Torus Chain - Sutherland interpolating Hamiltonian.
"""

import numpy as np

from tenpy.models.model import CouplingMPOModel
from models.networks.SutherlandSite import SutherlandSite


class Sutherland(CouplingMPOModel):

    def init_sites(self, model_params):
        "Import site definition"
        conserve = model_params.get("conserve", None)
        site = SutherlandSite(conserve)
        return site

    def init_terms(self, model_params):
        "Define hamiltonian"
        theta = model_params.get("theta", 1)
        w = np.exp(2.0j * np.pi / 3)

        for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
            self.add_coupling(np.cos(theta), u1, "s", u2, "sD", dx, plus_hc=True)
            self.add_coupling(np.cos(theta), u1, "t", u2, "tD", dx, plus_hc=True)
            self.add_coupling(w * np.sin(theta), u1, "stD", u2, "sDt", dx, plus_hc=True)
            self.add_coupling(w**2 * np.sin(theta), u1, "st", u2, "sDtD", dx, plus_hc=True)