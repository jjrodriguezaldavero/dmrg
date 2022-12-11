"""
Definition of the Axial Next-Nearest-Neighbor Ising sites.
"""

import numpy as np

from tenpy.networks.site import Site
from tenpy.linalg import np_conserved as npc


class ANNNISite(Site):

    def __init__(self, conserve="Q"):
        if conserve not in ["Q", None]:
            raise ValueError("invalid `conserve`: " + repr(conserve))

        sx = np.array([[0, 1], [1, 0]])
        sz = np.array([[1, 0], [0, 1]])
        
        ops = dict(sx=sx, sz=sz)

        N = 2
        Q = np.arange(N)
        self.q = N
        if conserve == "Q": 
            chinfo = npc.ChargeInfo([N], ["Q"])
            leg = npc.LegCharge.from_qflat(chinfo, Q)
        else:
            leg = npc.LegCharge.from_trivial(N)
        self.conserve = conserve

        names = [str(i) for i in np.arange(N)]
        Site.__init__(self, leg, names, **ops)

    def __repr__(self):
        return "ANNNISite(N={})".format(2)