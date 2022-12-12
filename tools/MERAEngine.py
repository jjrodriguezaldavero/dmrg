  
# doVarMERA.py
# ---------------------------------------------------------------------
# Variational energy minimization of (scale-invariant) modified binary MERA
#
# by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 6/2019

import numpy as np
from numpy import linalg as LA
import time
from scipy.sparse.linalg import eigs
from typing import List, Union, Tuple, Optional
import matplotlib.pyplot as plt

# import numba
# from numba import jit, config
# config.DISABLE_JIT = True

# from numba.core.errors import NumbaWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
# import logging
# numba_logger = logging.getLogger('numba')
# import warnings
# warnings.simplefilter('ignore', category=NumbaWarning)
# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)



def doVarMERA(tensors, E_tol, iters, chi, chimid, trlayers, sciter):
  """
  Variational energy minimization of (scale-invariant) modified binary MERA
  for nearest neighbour 1D Hamiltonian. Inputs 'hamAB, hamBA, rhoAB, rhoBA,
  wC, vC, uC' are lists whose lengths are equal to the number of MERA
  levels. Input Hamiltonian specified through 'hamAB[0]' and 'hamBA[0]'.
  Bond dimensions specified through 'chi' and 'chimid'.
  """

  refsym = True # Reflections symmetry
  verbose = True # Log out data
  converged = False # Convergence

  # Unpack tensors
  hamAB, hamBA = tensors['hamAB'], tensors['hamBA']
  rhoAB, rhoBA = tensors['rhoAB'], tensors['rhoBA']
  wC, vC, uC = tensors['wC'], tensors['vC'], tensors['uC']

  # Add extra layers if required
  totLv = trlayers + 1
  for k in range(totLv - len(wC)):
    wC.append(wC[-1])
    vC.append(vC[-1])
    uC.append(uC[-1])

  for k in range(1 + totLv - len(hamAB)):
    hamAB.append(hamAB[-1])
    hamBA.append(hamBA[-1])
    rhoAB.append(rhoAB[-1])
    rhoBA.append(rhoBA[-1])

  # Expand tensors to new dimensions if required
  chiZ = np.zeros(totLv + 1, dtype=int)
  chiZ[0] = hamAB[0].shape[0]
  chimidZ = np.zeros(totLv + 1, dtype=int)
  chimidZ[0] = hamAB[0].shape[0]
  for k in range(totLv):
    chiZ[k + 1] = min(chi, chiZ[k] * chimidZ[k])
    chimidZ[k + 1] = min(chimid, chiZ[k])
    wC[k] = TensorExpand(wC[k], [chiZ[k], chimidZ[k + 1], chiZ[k + 1]])
    vC[k] = TensorExpand(vC[k], [chiZ[k], chimidZ[k + 1], chiZ[k + 1]])
    uC[k] = TensorExpand(uC[k], [chiZ[k], chiZ[k], chimidZ[k + 1], chimidZ[k + 1]])
    hamAB[k + 1] = TensorExpand(hamAB[k + 1], [chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]])
    hamBA[k + 1] = TensorExpand(hamBA[k + 1], [chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]])
    rhoAB[k + 1] = TensorExpand(rhoAB[k + 1], [chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]])
    rhoBA[k + 1] = TensorExpand(rhoBA[k + 1], [chiZ[k + 1], chiZ[k + 1], chiZ[k + 1], chiZ[k + 1]])

  # Ensure Hamiltonian is negative defined
  hamABstart = hamAB[0]
  hamBAstart = hamBA[0]
  bias = max(LA.eigvalsh(hamAB[0].reshape(chiZ[0]**2, chiZ[0]**2)))
  hamAB[0] = hamAB[0] - bias * np.eye(chiZ[0]**2).reshape(chiZ[0], chiZ[0], chiZ[0], chiZ[0])
  hamBA[0] = hamBA[0] - bias * np.eye(chiZ[0]**2).reshape(chiZ[0], chiZ[0], chiZ[0], chiZ[0])

  times = np.zeros((10))
  energies = np.array([0])

  # Start loop
  for k in range(iters):
    tick = round(time.time(), 3)
    if np.mod(k, 10) == 0:
      timed = tick - times[0] if k != 0 else 0
    times[np.mod(k, 10)] = tick

    # Find scale-invariant density matrix (via power method)
    for g in range(sciter):
      rhoABtemp, rhoBAtemp = DescendSuper(rhoAB[totLv], rhoBA[totLv], wC[totLv - 1], vC[totLv - 1], uC[totLv - 1], refsym)
      rhoAB[totLv] = 0.5 * (rhoABtemp + np.conj(rhoABtemp.transpose(2, 3, 0, 1))) / ncon([rhoABtemp], [[1, 2, 1, 2]])
      rhoBA[totLv] = 0.5 * (rhoBAtemp + np.conj(rhoBAtemp.transpose(2, 3, 0, 1))) / ncon([rhoBAtemp], [[1, 2, 1, 2]])
      if refsym:
        rhoAB[totLv] = 0.5 * rhoAB[totLv] + 0.5 * rhoAB[totLv].transpose(1, 0, 3, 2)
        rhoBA[totLv] = 0.5 * rhoBA[totLv] + 0.5 * rhoBA[totLv].transpose(1, 0, 3, 2)

    # Descend density matrix through all layers
    for p in range(totLv - 1, -1, -1):
      rhoAB[p], rhoBA[p] = DescendSuper(rhoAB[p + 1], rhoBA[p + 1], wC[p], vC[p], uC[p], refsym)

    # Compute energy and display
    if verbose:
      if np.mod(k, 10) == 0:
        energy = (ncon([rhoAB[0], hamAB[0]], [[1, 2, 3, 4], [1, 2, 3, 4]]) + 
          ncon([rhoBA[0], hamBA[0]], [[1, 2, 3, 4], [1, 2, 3, 4]])) / 4 + bias / 2
        energy = energy.real
        energy_delta = energy - energies[-1]

        print('Chi %d | Iteration: %d of %d | Energy: %.8f | Delta_E: %.8f | Time: %.2f seconds' %
              (chi, k, iters, energy, energy_delta, timed))
        
        energies = np.append(energies, energy)

        if abs(energy_delta) <= E_tol:
          converged = True

    # Optimise over all layers
    for p in range(totLv):
      if k > 9:
        uEnv = DisEnv(hamAB[p], hamBA[p], rhoBA[p + 1], wC[p], vC[p], uC[p], refsym)
        if refsym:
          uEnv = uEnv + uEnv.transpose(1, 0, 3, 2)

        uC[p] = TensorUpdateSVD(uEnv, 2)

      if k > 1:
        wEnv = IsoEnvW(hamAB[p], hamBA[p], rhoBA[p + 1], rhoAB[p + 1], wC[p], vC[p], uC[p])
        wC[p] = TensorUpdateSVD(wEnv, 2)
        if refsym:
          vC[p] = wC[p]
        else:
          vEnv = IsoEnvV(hamAB[p], hamBA[p], rhoBA[p + 1], rhoAB[p + 1], wC[p], vC[p], uC[p])
          vC[p] = TensorUpdateSVD(vEnv, 2)

      hamAB[p + 1], hamBA[p + 1] = AscendSuper(hamAB[p], hamBA[p], wC[p], vC[p],uC[p], refsym)

    if converged == True: break

  hamAB[0] = hamABstart
  hamBA[0] = hamBAstart

  tensors = {'hamAB': hamAB, 'hamBA': hamBA, 'rhoAB': rhoAB, 'rhoBA': rhoBA, 'wC': wC, 'vC': vC, 'uC': uC}

  return energies, tensors, converged



def AscendSuper(hamAB, hamBA, w, v, u, refsym):
  """ apply the average ascending superoperator to the Hamiltonian """

  indList1 = [[6, 4, 1, 2], [1, 3, -3], [6, 7, -1], [2, 5, 3, 9], [4, 5, 7, 10], [8, 9, -4], [8, 10, -2]]
  indList2 = [[3, 4, 1, 2], [5, 6, -3], [5, 7, -1], [1, 2, 6, 9], [3, 4, 7, 10], [8, 9, -4], [8, 10, -2]]
  indList3 = [[5, 7, 2, 1], [8, 9, -3], [8, 10, -1], [4, 2, 9, 3], [4, 5, 10, 6], [1, 3, -4], [7, 6, -2]]
  indList4 = [[3, 6, 2, 5], [2, 1, -3], [3, 1, -1], [5, 4, -4], [6, 4, -2]]

  hamBAout = ncon([hamAB, w, np.conj(w), u, np.conj(u), v, np.conj(v)], indList1)
  if refsym:
    hamBAout = hamBAout + hamBAout.transpose(1, 0, 3, 2)
  else:
    hamBAout = hamBAout + ncon([hamAB, w, np.conj(w), u, np.conj(u), v, np.conj(v)], indList3)

  hamBAout = hamBAout + ncon([hamBA, w, np.conj(w), u, np.conj(u), v, np.conj(v)], indList2)
  hamABout = ncon([hamBA, v, np.conj(v), w, np.conj(w)], indList4)

  return hamABout, hamBAout



def DescendSuper(rhoAB, rhoBA, w, v, u, refsym):
  """ apply the average descending superoperator to the density matrix """

  indList1 = [[9, 3, 4, 2], [-3, 5, 4], [-1, 10, 9], [-4, 7, 5, 6], [-2, 7, 10, 8], [1, 6, 2], [1, 8, 3]]
  indList2 = [[3, 6, 2, 5], [1, 7, 2], [1, 9, 3], [-3, -4, 7, 8], [-1, -2, 9, 10], [4, 8, 5], [4, 10, 6]]
  indList3 = [[3, 9, 2, 4], [1, 5, 2], [1, 8, 3], [7, -3, 5, 6], [7, -1, 8, 10], [-4, 6, 4], [-2, 10, 9]]
  indList4 = [[3, 6, 2, 5], [-3, 1, 2], [-1, 1, 3], [-4, 4, 5], [-2, 4, 6]]

  rhoABout = 0.5 * ncon([rhoBA, w, np.conj(w), u, np.conj(u), v, np.conj(v)], indList1)
  if refsym:
    rhoABout = rhoABout + rhoABout.transpose(1, 0, 3, 2)
  else:
    rhoABout = rhoABout + 0.5 * ncon([rhoBA, w, np.conj(w), u, np.conj(u), v, np.conj(v)], indList3)

  rhoBAout = 0.5 * ncon([rhoBA, w, np.conj(w), u, np.conj(u), v, np.conj(v)], indList2)
  rhoBAout = rhoBAout + 0.5 * ncon([rhoAB, v, np.conj(v), w, np.conj(w)], indList4)

  return rhoABout, rhoBAout



def DisEnv(hamAB, hamBA, rhoBA, w, v, u, refsym):
  """ compute the environment of a disentangler """

  indList1 = [[7, 8, 10, -1], [4, 3, 9, 2], [10, -3, 9], [7, 5, 4], [8, -2, 5, 6], [1, -4, 2], [1, 6, 3]]
  indList2 = [[7, 8, -1, -2], [3, 6, 2, 5], [1, -3, 2], [1, 9, 3], [7, 8, 9, 10], [4, -4, 5], [4, 10, 6]]
  indList3 = [[7, 8, -2, 10], [3, 4, 2, 9], [1, -3, 2], [1, 5, 3], [-1, 7, 5, 6], [10, -4, 9], [8, 6, 4]]

  uEnv = ncon([hamAB, rhoBA, w, np.conj(w), np.conj(u), v, np.conj(v)], indList1)
  if refsym:
    uEnv = uEnv + uEnv.transpose(1, 0, 3, 2)
  else:
    uEnv = uEnv + ncon([hamAB, rhoBA, w, np.conj(w), np.conj(u), v, np.conj(v)], indList3)

  uEnv = uEnv + ncon([hamBA, rhoBA, w, np.conj(w), np.conj(u), v, np.conj(v)], indList2)

  return uEnv



def IsoEnvW(hamAB, hamBA, rhoBA, rhoAB, w, v, u):
  """ compute the environment of a 'w'-isometry """

  indList1 = [[7, 8, -1, 9], [4, 3, -3, 2], [7, 5, 4], [9, 10, -2, 11], [8, 10, 5, 6], [1, 11, 2], [1, 6, 3]]
  indList2 = [[1, 2, 3, 4], [10, 7, -3, 6], [-1, 11, 10], [3, 4, -2, 8], [1, 2, 11, 9], [5, 8, 6], [5, 9, 7]]
  indList3 = [[5, 7, 3, 1], [10, 9, -3, 8], [-1, 11, 10], [4, 3, -2, 2], [4, 5, 11, 6], [1, 2, 8], [7, 6, 9]]
  indList4 = [[3, 7, 2, -1], [5, 6, 4, -3], [2, 1, 4], [3, 1, 5], [7, -2, 6]]

  wEnv = ncon([hamAB, rhoBA, np.conj(w), u, np.conj(u), v, np.conj(v)], indList1)
  wEnv = wEnv + ncon([hamBA, rhoBA, np.conj(w), u, np.conj(u), v, np.conj(v)], indList2)
  wEnv = wEnv + ncon([hamAB, rhoBA, np.conj(w), u, np.conj(u), v, np.conj(v)], indList3)
  wEnv = wEnv + ncon([hamBA, rhoAB, v, np.conj(v), np.conj(w)], indList4)

  return wEnv



def IsoEnvV(hamAB, hamBA, rhoBA, rhoAB, w, v, u):
  """ compute the environment of a 'v'-isometry """

  indList1 = [[6, 4, 1, 3], [9, 11, 8, -3], [1, 2, 8], [6, 7, 9], [3, 5, 2, -2], [4, 5, 7, 10], [-1, 10, 11]]
  indList2 = [[3, 4, 1, 2], [8, 10, 9, -3], [5, 6, 9], [5, 7, 8], [1, 2, 6, -2], [3, 4, 7, 11], [-1, 11, 10]]
  indList3 = [[9, 10, 11, -1], [3, 4, 2, -3], [1, 8, 2], [1, 5, 3], [7, 11, 8, -2], [7, 9, 5, 6], [10, 6, 4]]
  indList4 = [[7, 5, -1, 4], [6, 3, -3, 2], [7, -2, 6], [4, 1, 2], [5, 1, 3]]

  vEnv = ncon([hamAB, rhoBA, w, np.conj(w), u, np.conj(u), np.conj(v)], indList1)
  vEnv = vEnv + ncon([hamBA, rhoBA, w, np.conj(w), u, np.conj(u), np.conj(v)], indList2)
  vEnv = vEnv + ncon([hamAB, rhoBA, w, np.conj(w), u, np.conj(u), np.conj(v)], indList3)
  vEnv = vEnv + ncon([hamBA, rhoAB, np.conj(v), w, np.conj(w)], indList4)

  return vEnv



def TensorExpand(A, chivec):
  """ expand tensor dimension by padding with zeros """

  if [*A.shape] == chivec:
    return A
  else:
    for k in range(len(chivec)):
      if A.shape[k] != chivec[k]:
        indloc = list(range(-1, -len(chivec) - 1, -1))
        indloc[k] = 1
        A = ncon([A, np.eye(A.shape[k], chivec[k])], [indloc, [1, -k - 1]])

    return A



def TensorUpdateSVD(wIn, leftnum):
  """ update an isometry using its (linearized) environment """

  wSh = wIn.shape
  ut, st, vht = LA.svd(
      wIn.reshape(np.prod(wSh[0:leftnum:1]), np.prod(wSh[leftnum:len(wSh):1])),
      full_matrices=False)
  return -(ut @ vht).reshape(wSh)



def doConformalMERA(wS, uS, vS, rhoBAS, scnum):
  """
  Compute conformal data from an modified binary MERA optimized for a
  scale-invariant critical point. Input 'ws', 'vs' and 'uS' are the
  isometries and disentangler from the scale-invariant layers, while
  'rhoBAS' is the fixed point density matrix. 'scnum' sets the number of
  scaling dimensions to compute.

  Outputs 'scDims', 'scOps', and 'Cfusion' are the scaling dimensions,
  scaling operators and fusion coefficients respectively.
  """

  # Diagonalize 1-site scaling superoperator
  chi = wS.shape[2]
  tensors = [wS, np.conj(wS), vS, np.conj(vS)]
  connects = [[-4, 1, 3], [-3, 1, 4], [3, 2, -2], [4, 2, -1]]
  ScSuper1 = ncon(tensors, connects).reshape(chi**2, chi**2)

  dtemp, utemp = eigs(ScSuper1, k=scnum, which='LM')
  scDims = -np.log2(abs(dtemp)) / 2

  # Normalize scaling operators
  scOps = [0 for x in range(scnum)]
  for k in range(scnum):
    scAtemp = utemp[:, k].reshape(chi, chi)
    scAtemp = scAtemp / LA.norm(scAtemp)

    tensors = [scAtemp, scAtemp, wS, np.conj(wS), uS, np.conj(uS), vS, np.conj(vS), rhoBAS]
    connects = [[8, 7], [3, 1], [7, 9, 11], [8, 10, 13], [2, 1, 9, 5], [2, 3, 10, 6], [4, 5, 12], [4, 6, 14], [13, 14, 11, 12]]
    cweight = ncon(tensors, connects)
    scOps[k] = scAtemp / np.sqrt(cweight)

  # Compute fusion coefficients (OPE coefficients)
  Cfusion = np.zeros((scnum, scnum, scnum), dtype=complex)
  for k1 in range(scnum):
    for k2 in range(scnum):
      for k3 in range(scnum):
        Otemp = scDims[k1] - scDims[k2] + scDims[k3]
        tensors = [scOps[k1], scOps[k2], scOps[k3], wS, np.conj(wS), uS,
                   np.conj(uS), vS, np.conj(vS), uS, np.conj(uS), wS,
                   np.conj(wS), wS, np.conj(wS), vS, np.conj(vS), rhoBAS]
        connects = [[5, 4], [3, 1], [28, 27], [4, 6, 11], [5, 7, 13],
                    [2, 1, 6, 9], [2, 3, 7, 10], [8, 9, 12], [8, 10, 14],
                    [11, 12, 16, 21], [13, 14, 17, 23], [15, 16, 18],
                    [15, 17, 19], [27, 26, 24], [28, 26, 25], [24, 21, 20],
                    [25, 23, 22], [19, 22, 18, 20]]
        Cfusion[k1, k2, k3] = (2**Otemp) * ncon(tensors, connects)

  return scDims, scOps, Cfusion


def ncon(tensors,#: List[np.ndarray],
         connects,#: List[Union[List[int], Tuple[int]]],
         con_order=None,#: Optional[Union[List[int], str]] = None,
         check_network=False,#: Optional[bool] = True,
         which_env=0):#: Optional[int] = 0):
  """
  Network CONtractor: contracts a tensor network of N tensors via a sequence
  of (N-1) tensordot operations. More detailed instructions and examples can
  be found at: https://arxiv.org/abs/1402.0939.
  Args:
    tensors: list of the tensors in the network.
    connects: length-N list of lists (or tuples) specifying the network
      connections. The jth entry of the ith list in connects labels the edge
      connected to the jth index of the ith tensor. Labels should be positive
      integers for internal indices and negative integers for free indices.
    con_order: optional argument to specify the order for contracting the
      positive indices. Defaults to ascending order if omitted. Can also be
      set at "greedy" or "full" to call a solver to automatically determine
      the order.
    check_network: if true then the input network is checked for consistency;
      this can catch many common user mistakes for defining networks.
    which_env: if provided, ncon will produce the environment of the requested
      tensor (i.e. the network given by removing the specified tensor from
      the original network). Only valid for networks with no open indices.
  Returns:
    Union[np.ndarray,float]: the result of the network contraction; an
      np.ndarray if the network contained open indices, otherwise a scalar.
  """
  num_tensors = len(tensors)
  tensor_list = [tensors[ele] for ele in range(num_tensors)]
  # tensor_list = np.array([], dtype=numba.float64)
  # for ele in range(num_tensors):
  #   tensor_list = np.append(tensor_list, tensors[ele])
  connect_list = [np.array(connects[ele]) for ele in range(num_tensors)]

  # generate contraction order if necessary
  flat_connect = np.concatenate(connect_list)
  if con_order is None:
    con_order = np.unique(flat_connect[flat_connect > 0])
  else:
    con_order = np.array(con_order)

  # check inputs if enabled
  # if check_network:
  #   dims_list = [list(tensor.shape) for tensor in tensor_list]
  #   check_inputs(connect_list, flat_connect, dims_list, con_order)

  # do all partial traces
  for ele in range(len(tensor_list)):
    num_cont = len(connect_list[ele]) - len(np.unique(connect_list[ele]))
    if num_cont > 0:
      tensor_list[ele], connect_list[ele], cont_ind = partial_trace(
          tensor_list[ele], connect_list[ele])
      con_order = np.delete(
          con_order,
          np.intersect1d(con_order, cont_ind, return_indices=True)[1])

  # do all binary contractions
  while len(con_order) > 0:
    # identify tensors to be contracted
    cont_ind = con_order[0]
    locs = [
        ele for ele in range(len(connect_list))
        if sum(connect_list[ele] == cont_ind) > 0
    ]

    # do binary contraction
    cont_many, A_cont, B_cont = np.intersect1d(
        connect_list[locs[0]],
        connect_list[locs[1]],
        assume_unique=True,
        return_indices=True)
    if np.size(tensor_list[locs[0]]) < np.size(tensor_list[locs[1]]):
      ind_order = np.argsort(A_cont)
    else:
      ind_order = np.argsort(B_cont)

    tensor_list.append(
        np.tensordot(
            tensor_list[locs[0]],
            tensor_list[locs[1]],
            axes=(A_cont[ind_order], B_cont[ind_order])))
    connect_list.append(
        np.append(
            np.delete(connect_list[locs[0]], A_cont),
            np.delete(connect_list[locs[1]], B_cont)))

    # remove contracted tensors from list and update con_order
    del tensor_list[locs[1]]
    del tensor_list[locs[0]]
    del connect_list[locs[1]]
    del connect_list[locs[0]]
    con_order = np.delete(
        con_order,
        np.intersect1d(con_order, cont_many, return_indices=True)[1])

  # do all outer products
  while len(tensor_list) > 1:
    s1 = tensor_list[-2].shape
    s2 = tensor_list[-1].shape
    tensor_list[-2] = np.outer(tensor_list[-2].reshape(np.prod(s1)),
                               tensor_list[-1].reshape(np.prod(s2))).reshape(
                                   np.append(s1, s2))
    connect_list[-2] = np.append(connect_list[-2], connect_list[-1])
    del tensor_list[-1]
    del connect_list[-1]

  # do final permutation
  if len(connect_list[0]) > 0:
    return np.transpose(tensor_list[0], np.argsort(-connect_list[0]))
  else:
    return tensor_list[0].item()

def partial_trace(A, A_label):
  """ Partial trace on tensor A over repeated labels in A_label """

  num_cont = len(A_label) - len(np.unique(A_label))
  if num_cont > 0:
    dup_list = []
    for ele in np.unique(A_label):
      if sum(A_label == ele) > 1:
        dup_list.append([np.where(A_label == ele)[0]])

    cont_ind = np.array(dup_list).reshape(2 * num_cont, order='F')
    free_ind = np.delete(np.arange(len(A_label)), cont_ind)

    cont_dim = np.prod(np.array(A.shape)[cont_ind[:num_cont]])
    free_dim = np.array(A.shape)[free_ind]

    B_label = np.delete(A_label, cont_ind)
    cont_label = np.unique(A_label[cont_ind])
    B = np.zeros(np.prod(free_dim))
    A = A.transpose(np.append(free_ind, cont_ind)).reshape(
        np.prod(free_dim), cont_dim, cont_dim)
    for ip in range(cont_dim):
      B = B + A[:, ip, ip]

    return B.reshape(free_dim), B_label, cont_label

  else:
    return A, A_label, []


def check_inputs(connect_list, flat_connect, dims_list, con_order):
  """ Check consistancy of NCON inputs"""

  pos_ind = flat_connect[flat_connect > 0]
  neg_ind = flat_connect[flat_connect < 0]

  # check that lengths of lists match
  if len(dims_list) != len(connect_list):
    raise ValueError(
        ('mismatch between %i tensors given but %i index sublists given') %
        (len(dims_list), len(connect_list)))

  # check that tensors have the right number of indices
  for ele in range(len(dims_list)):
    if len(dims_list[ele]) != len(connect_list[ele]):
      raise ValueError((
          'number of indices does not match number of labels on tensor %i: '
          '%i-indices versus %i-labels')
          % (ele, len(dims_list[ele]), len(connect_list[ele])))

  # check that contraction order is valid
  if not np.array_equal(np.sort(con_order), np.unique(pos_ind)):
    raise ValueError(('NCON error: invalid contraction order'))

  # check that negative indices are valid
  for ind in np.arange(-1, -len(neg_ind) - 1, -1):
    if sum(neg_ind == ind) == 0:
      raise ValueError(('NCON error: no index labelled %i') % (ind))
    elif sum(neg_ind == ind) > 1:
      raise ValueError(('NCON error: more than one index labelled %i') % (ind))

  # check that positive indices are valid and contracted tensor dimensions match
  flat_dims = np.array([item for sublist in dims_list for item in sublist])
  for ind in np.unique(pos_ind):
    if sum(pos_ind == ind) == 1:
      raise ValueError(('NCON error: only one index labelled %i') % (ind))
    elif sum(pos_ind == ind) > 2:
      raise ValueError(
          ('NCON error: more than two indices labelled %i') % (ind))

    cont_dims = flat_dims[flat_connect == ind]
    if cont_dims[0] != cont_dims[1]:
      raise ValueError(
          ('NCON error: tensor dimension mismatch on index labelled %i: '
           'dim-%i versus dim-%i') % (ind, cont_dims[0], cont_dims[1]))

  return True