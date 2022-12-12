"""
Library with auxiliary functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from tenpy.algorithms.exact_diag import ExactDiag
from algorithms import DMRG, MERA

def find_nearest(array, value):
    """
    Finds the index of the nearest value to a given value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def crop_array(array, ranges, fixed_values):
    """
    Crops an array given two fixed values.
    """
    if fixed_values is not None:
        names = list(ranges.keys()); indices = [0, 1, 2, 3]

        name_indices = [names.index(fixed_values['name_1']), names.index(fixed_values['name_2'])]
        value_indices = [np.where(np.asarray(ranges[fixed_values['name_1']]) == float(fixed_values['value_1']))[0][0], 
                        np.where(np.asarray(ranges[fixed_values['name_2']]) == float(fixed_values['value_2']))[0][0]]
        
        indices.pop(name_indices[1]); indices.pop(name_indices[0])

        cropped_array = np.transpose(np.take(np.take(array, value_indices[1], name_indices[1]), value_indices[0], name_indices[0]))
        range_names = [names[indices[1]], names[indices[0]]]
        range_values = [ranges[names[indices[1]]], ranges[names[indices[0]]]]

        return cropped_array, range_names, range_values 

    else:
        range_names = list(reversed(ranges.keys()))
        range_values = list(reversed(ranges.values()))
        return array.T, range_names, range_values


def compute_critical_exponent(observable, ranges, fixed_values, crit_val):
    """
    Fits the critical exponent.
    Find slope of log(obs) against log(L) at the critical point
    """
    #ind_crit = np.where(crit_range == crit_val)[0][0]
    cropped_array, _, range_values = crop_array(observable, ranges, fixed_values)
    ind_crit = find_nearest(range_values[0], crit_val)
    obs_crit = abs(cropped_array[ind_crit])

    fit = np.polyfit(np.log(range_values[1]), np.log(obs_crit), 1)
    exponent = fit[0]
    constant = fit[1]

    return exponent, constant


def compute_fss_range(central_value, n=8, delta=0.004):
    """
    Computes a range of values given the central value.
    """
    values = []
    for i in range(2 * n + 1):
        vi = round(central_value - delta + i * delta / n, 4) # Rounds to kill machine error
        values.append(vi)

    return np.array(values)


def compute_critical_points(points, ranges, threshold):
    """
    Return a list of all points that are above a certain threshold
    """
    critical_indices = np.where(points > threshold)

    F_values = np.array(ranges['F'])[critical_indices[1]]
    U_values = np.array(ranges['U'])[critical_indices[2]]
    V_values = np.array(ranges['V'])[critical_indices[3]]

    return list(zip(F_values, U_values, V_values))


def compute_peschel_emery(r):
    """
    Computes the Peschel-Emery line as a function of parameter r > 0
    """

    U = 2 * (1 - r)**2 * (1 + r + r**2) / (9*r**2)
    F = 2 * (1 + 2*r) * (1 - r**3) / (9*r**2)
    V = -(1 - r)**2 * (1 - 2*r - 2*r**2) / (9*r**2)

    return {'U': U, 'F': F, 'V': V}


def build_MERA_from_Model(Model, model_params, d):
    """
    Builds a hamiltonian suitable for MERA from the two-site periodic hamiltonian 
    """
    model_params.update({'conserve': None, 'bc_MPS': 'finite', 'bc_x': 'periodic', })
    diag = ExactDiag(Model(model_params))
    diag.build_full_H_from_mpo()

    # Transform from two to four sites
    H_two = diag.full_H.to_ndarray()
    H_four = (np.kron(np.eye(d), np.kron(H_two, np.eye(d))) + (1/2) * (np.kron(np.eye(d**2), H_two) + np.kron(H_two, np.eye(d**2)))).reshape(d,d,d,d,d,d,d,d)
    return H_four


def build_worker_DMRG(Model, model_params, dmrg_params, sector_params, simulation_path):
    """
    Builds a worker for DMRG and a given model.
    """
    if Model.__module__ == "models.ANNNP":
        parameters = ['L', 'F', 'U', 'V']
        q = 3
        correlation_operators = ("s", "sD")

    elif Model.__module__ == "models.Sutherland":
        parameters = ['L', 'theta']
        q = 3
        correlation_operators = ("s", "sD")

    elif Model.__module__ == "models.ANNNI":
        parameters = ['L', 'D', 'U', 'E']
        q = 2
        correlation_operators = ("sz", "sz")

    # Add more models (Hamiltonian parameters, site dimension and correlation operators)

    def worker(value):
        H_params = dict(zip(parameters, value))
        name = ''.join('{}{}_'.format(key, round(val, 4)) for key, val in H_params.items())[:-1]
        data_path = simulation_path + "data/"
        os.makedirs(data_path, exist_ok=True)
        try:
            print("Trying to load {}".format(name))
            with open(data_path + name, 'rb') as f:
                pickle.load(f)
        except:
            model_params.update(H_params)
            model = Model(model_params)
            data, _ = DMRG.run(dmrg_params, model, sector_params, q=q, correlation_operators=correlation_operators)
            with open(data_path + name, 'wb+') as f:
                pickle.dump(data, f, 2)

    return worker


def build_worker_MERA(Model, mera_params, simulation_path, use_checkpoint=False):
    """
    Builds a worker for MERA and a given model.
    """
    if Model.__module__ == "models.ANNNP":
        parameters = ['F', 'U', 'V']
        d = 3

    elif Model.__module__ == "models.Sutherland":
        parameters = ['theta']
        d = 3

    elif Model.__module__ == "models.ANNNI":
        parameters = ['D', 'U', 'E']
        d = 2

    def worker(value):
        H_params = dict(zip(parameters, value))
        name = ''.join('{}{}_'.format(key, round(val, 4)) for key, val in H_params.items())[:-1]
        data_path = simulation_path + "data/"
        os.makedirs(data_path, exist_ok=True)
        model = build_MERA_from_Model(Model, H_params, d)

        try:
            print("Trying to load {}".format(name))
            with open(data_path + name, 'rb') as f:
                _, saved_tensors = pickle.load(f)
            if use_checkpoint:
                point, tensors = MERA.run(mera_params, model, saved_tensors)
                # Match parameters with saved data
                mera_params['chi_init'] = point['checkpoint_data']['chi']
                mera_params['iters_init'] = point['checkpoint_data']['iters']
                mera_params['layers_init'] = point['checkpoint_data']['layers']
                with open(data_path + name, 'wb+') as f:
                    pickle.dump((point, tensors), f, 2)
        except:
            point, tensors = MERA.run(mera_params, model)
            with open(data_path + name, 'wb+') as f:
                pickle.dump((point, tensors), f, 2)

    return worker


def synchronize(home_path, simulation_path, sync_direction):
    SSH_DIR = "6835384@gemini.science.uu.nl"
    home_path_cut = home_path.replace(home_path.split('/')[-1], '') 
    simulation_path_cut = simulation_path.replace(home_path_cut, '') + '/'

    if sync_direction == "DOWNSTREAM":
        os.system("rsync -avhP " + SSH_DIR + ":" + simulation_path_cut + "data/ " + simulation_path + "/data/")  
    elif sync_direction == "UPSTREAM":
        os.system("rsync -avhP " + simulation_path + "/data/ " + SSH_DIR + ":" + simulation_path_cut + "data/")  
    elif sync_direction == "UPSTREAM DELETE":
        os.system("rsync -avhP --delete " + simulation_path + "/data/ " + SSH_DIR + ":" + simulation_path_cut + "data/")  
