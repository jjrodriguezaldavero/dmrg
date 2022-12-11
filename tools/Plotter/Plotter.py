"""
Library with common plotting routines for many models.
"""

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tools import tools as t


class Plotter():

    def __init__(self, H_params, simulation_path):
        self.H_params = H_params
        self.simulation_path = simulation_path


    def plot_central_charges(self, charges, fixed_values, name_suffix='', plot_params={}):
        """
        Plots the central charges with respect to two fixed values.
        """    
        # Crops the multidimensional array to consider the two fixed values.
        cropped_array, range_names, range_values = t.crop_array(charges, self.H_params, fixed_values)
        
        # Plots the cropped array using Seaborn
        sns.set_style('whitegrid')
        plt.plot(cropped_array, '-1')
        plt.legend(range_values[1], title=range_names[1])

        # Read plot parameters
        x_tick_periodicity = 2
        ticks = plot_params['ticks'] if 'ticks' in plot_params else (np.arange(
            0, len(range_values[0]), x_tick_periodicity), np.around(range_values[0], 2)[0::x_tick_periodicity])
        title = plot_params['title'] if 'title' in plot_params else r'Finite-size scaling for the central charge $c(L)$'
        xlabel = plot_params['xlabel'] if 'xlabel' in plot_params else r"${}$".format(range_names[0])
        ylabel = plot_params['ylabel'] if 'ylabel' in plot_params else 'Central charge c'

        plt.xticks(ticks[0], ticks[1])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.savefig(self.simulation_path + 'figures/central_charges' + name_suffix)
        plt.show()


    def plot_finite_size(self, observable, critical_exponent, fixed_values, title, ylabel, path_figure, plot_params):
        """
        Auxiliar function to plot observables that show finite size effects with a critical exponent.
        """
        # Crops the multidimensional array to consider the two fixed values.
        cropped_array, range_names, range_values = t.crop_array(observable, self.H_params, fixed_values)

        # Plots the cropped array using Seaborn
        sns.set_style('whitegrid')
        plt.plot(cropped_array * range_values[1] ** critical_exponent, '-1')
        plt.legend(range_values[1], title=range_names[1])

        # Read plot parameters
        x_tick_periodicity = 2
        ticks = plot_params['ticks'] if 'ticks' in plot_params else np.arange(
            0, len(range_values[0]), x_tick_periodicity), np.around(range_values[0], 4)[0::x_tick_periodicity]
        title = plot_params['title'] if 'title' in plot_params else title
        xlabel = plot_params['xlabel'] if 'xlabel' in plot_params else r"${}$".format(range_names[0])
        ylabel = plot_params['ylabel'] if 'ylabel' in plot_params else ylabel

        plt.xticks(*ticks)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        x_tick_periodicity = 2
        plt.xticks(np.arange(0, len(range_values[0]), x_tick_periodicity), np.around(range_values[0], 4)[0::x_tick_periodicity])
        
        plt.title(title)
        plt.xlabel(r"${}$".format(range_names[0]))
        plt.ylabel(ylabel)

        plt.savefig(path_figure)
        plt.show()


    def plot_finite_size_gaps(self, gaps, fixed_values, crit_x=None, n_gap=0, name_suffix='', plot_params={}):
        """
        Plots the finite size effects for the energy gaps. The critical x-value must be specified in advance.
        """
        # Computes the critical exponent by fitting the energy gaps at the critical point.
        if crit_x is not None:
            critical_exponent, _ = t.compute_critical_exponent(gaps[n_gap], self.H_params, fixed_values, crit_x)
            z = round(-critical_exponent, 3)
            title = r'Finite-size scaling for the energy gap $\Delta$ for $z = {}$'.format(z)
            ylabel = r'$L^{{{}}}\cdot\Delta$'.format(round(-critical_exponent, 3))
        else:
            critical_exponent = 0
            title = r'Energy gap $\Delta$'
            ylabel = r'$L$'

        # Plots the energy gaps
        path_figure = self.simulation_path + 'figures/finite_size_gaps' + name_suffix
        self.plot_finite_size(gaps[n_gap], -critical_exponent, fixed_values, title, ylabel, path_figure, plot_params)


    def plot_finite_size_betas(self, betas, fixed_values, crit_x=None, name_suffix='', plot_params={}):
        """
        Plots the finite size effects for the Callan-Symanzik beta functions. The critical x-value must be specified in advance.
        """
        # Computes the critical exponent by fitting the energy gaps at the critical point.
        if crit_x is not None:
            critical_exponent, _ = t.compute_critical_exponent(abs(betas), self.H_params, fixed_values, crit_x)
            nu = -critical_exponent
            title = r'Finite-size scaling for the Callan-Symanzik $\beta$ function for $\nu = {}$'.format(round(nu, 3))
            ylabel = r'$L^{{{}}}\cdot\beta$'.format(round(-critical_exponent, 3))
        else:
            critical_exponent = 0
            title = r'Callan-Symanzik $\beta$ function'
            ylabel = r'$L$'

        # Plots the Callan-Symanzik betas
        path_figure = self.simulation_path + 'figures/finite_size_betas' + name_suffix
        self.plot_finite_size(betas, -critical_exponent, fixed_values, title, ylabel, path_figure, plot_params)


    def plot_finite_size_correlations(self, correlations, fixed_values, crit_x=None, name_suffix='', plot_params={}):
        """
        Plots the finite size effects for the spin-spin correlation functions. The critical x-value must be specified in advance.
        """
        # Computes the critical exponent by fitting the energy gaps at the critical point.
        if crit_x is not None:
            critical_exponent, _ = t.compute_critical_exponent(correlations, self.H_params, fixed_values, crit_x)
            eta = round(2 - critical_exponent, 3)
            title = r'Finite-size scaling for the two-point correlator $S(L)$ with $2 - \eta = {}$'.format(round(2 - eta, 3))
            ylabel = r'$L^{{{}}}\cdot S(L)$'.format(round(-critical_exponent, 3))
        else:
            critical_exponent = 0
            title = r'Two-point correlator $S(L)$'
            ylabel = r'$S(L)$'

        # Plots the two point correlations
        path_figure = self.simulation_path + 'figures/finite_size_correlations' + name_suffix
        self.plot_finite_size(correlations, -critical_exponent, fixed_values, title, ylabel, path_figure, plot_params)