"""
Definition of the Axial Next-Nearest-Neighbor Potts sites.
"""

import numpy as np

from tenpy.networks.site import Site
from tenpy.linalg import np_conserved as npc


class ANNNPSite(Site):

    def __init__(self, conserve="Q"):
        if conserve not in ["Q", None]:
            raise ValueError("invalid `conserve`: " + repr(conserve))

        N=3 # Generalizable to larger N
        t = np.zeros([N, N], dtype=np.complex64)
        for i in range(N):
            t[i][i] = np.exp(2.0j * i * np.pi / N)
        tD = t.conj().transpose()

        s = np.zeros([N, N])
        for i in range(N):
            iP = (i + 1) % N
            s[i][iP] = 1
        sD = s.conj().transpose()
        
        ops = dict(t=t, tD=tD, s=s, sD=sD)

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
        return "ANNNPSite(N={})".format(self.N)