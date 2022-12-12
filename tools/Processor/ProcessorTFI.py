"""
Class that processes a folder with data and creates multidimensional arrays of observables for the TFI model.
"""

import numpy as np
from math import log, pi
from scipy.optimize import curve_fit

import pickle
from itertools import product

class ProcessorTFI():
    """
    Class that contains all the processing functions needed for DMRG data.
    """
    def __init__(self, H_params, sector_params, simulation_path):
        self.H_params = H_params
        self.sector_params = sector_params
        self.simulation_path = simulation_path


    def build_array(self):
        """
        Initializes and populates arrays containing DMRG observables for several simulation points.
        """
        # Initializes empty arrays with the dimension of the simulation points cartesian product.
        dimensions = (len(self.H_params['L']), len(self.H_params['J']), len(self.H_params['g']))
        array_energies = np.zeros(dimensions, dtype=object)
        array_entropies = np.zeros(dimensions, dtype=object)
        array_correlations = np.zeros(dimensions, dtype=object)
        array_convergences = np.zeros(dimensions, dtype=object)

        def enumerated_product(*args):
            """
            Helper function for accessing both the index and value from the simulation points cartesian product.
            """
            yield from zip(product(*(range(len(x)) for x in args)), product(*args))
        
        # Loads or computes DMRG observables from each simulation point into the empty arrays and returns them in a dictionary.
        for index, values in enumerated_product(self.H_params['L'], self.H_params['J'], self.H_params['g']):
            value = {'L': float(values[0]), 'J': float(values[1]), 'g': float(values[2])}
            name = ''.join('{}{}_'.format(key, round(val, 4)) for key, val in value.items())[:-1]
            try:
                print("Trying to load {}".format(name))
                with open(self.simulation_path + 'data/' + name, 'rb') as f:
                    point = pickle.load(f)
            except:
                break

            array_energies[index] = point['energies']
            array_entropies[index] = point['entropy']
            array_correlations[index] = point['correlation']
            array_convergences[index] = point['convergences']

        self.array = {
            "energies": array_energies, 
            "entropies": array_entropies, 
            "correlations": array_correlations,
            "convergences": array_convergences
        }

        return self.array

        
    def compute_central_charges(self):
        """
        Computes an array of central charges from an array of entanglement entropies using the analytical formula.
        """
        try:
            entropies = self.array["entropies"]
        except:
            entropies = self.build_array()["entropies"]

        charges = np.zeros((entropies.shape[0] - 1, entropies.shape[1], entropies.shape[2]))
        for index in product(range(charges.shape[0]), range(charges.shape[1]), range(charges.shape[2])):
            entropy1 = entropies[index]
            entropy_max = entropies[-1][index[1:]]
            L = len(entropy1)
            Lmax = len(entropy_max)
            charge = 6 * (entropy1[L//2] - entropy_max[Lmax//2]) / log(L/Lmax)
            charges[index] = charge

        return charges

    
    def compute_central_charges_fit(self, end_cut=1):
        """
        Computes an array of central charges from an array of entanglement entropies by fitting the Calabrese-Cardy formula.
        The parameter end_cut eliminates some entropies at the edges for better convergence.
        """
        try:
            entropies = self.array["entropies"]
        except:
            entropies = self.build_array()["entropies"]

        def calabrese_cardy(L, l, S0, c):
            """
            Calabrese-Cardy formula for the entanglement entropy of a conformally invariant system.
            """
            return S0 + c/6 * np.log(L/pi * np.sin(pi * l/L))

        charges = np.zeros((entropies.shape[0], entropies.shape[1], entropies.shape[2]))
        for index in product(range(charges.shape[0]), range(charges.shape[1]), range(charges.shape[2])):
            L = entropies[index].shape[0]
            xdata = np.array(range(entropies[index].shape[0])[end_cut:-end_cut], dtype=np.float64)
            ydata = entropies[index][end_cut:-end_cut]
            def helper(l, S0, charge):
                return calabrese_cardy(L, l, S0, charge)
            params, _ = curve_fit(helper, xdata, ydata)
            charges[index] = abs(params[1])

        return charges


    def compute_gaps(self):
        """
        Computes an array of energy gaps from an array of individual energies.
        """
        try:
            energies = self.array["energies"]
        except:
            energies = self.build_array()["energies"]

        gaps = np.zeros((self.sector_params["n_states"] - 1, energies.shape[0], energies.shape[1], energies.shape[2]))
        for index in product(range(gaps.shape[0]), range(gaps.shape[1]), range(gaps.shape[2]), range(gaps.shape[3])):
            gaps[index] = (energies[index[1:]])[index[0] + 1] - (energies[index[1:]])[index[0]]
        return gaps


    def compute_correlations(self):
        """
        Computes an array of spin-spin correlation functions by summing up DMRG correlation data.
        """
        try:
            correlations = self.array["correlations"]
        except:
            correlations = self.build_array()["correlations"]

        correlators = np.zeros((correlations.shape[0], correlations.shape[1], correlations.shape[2]))
        for index in product(range(correlations.shape[0]), range(correlations.shape[1]), range(correlations.shape[2])):
            correlators[index] = np.sum(correlations[index])
        return correlators