"""
Library with plotting routines specific for the interpolating Hamiltonian between
the Quantum Torus Chain and the Sutherland model.
"""

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tools.Plotter.Plotter import Plotter
from tools import tools as t


class PlotterSutherland(Plotter):

    def plot_spectrum(self, energies):
        """
        Plots the energy spectrum.
        """

        fig = plt.figure()
        fig.suptitle("Energy spectrum")

        for L in range(energies.shape[0]):
            energy = energies[L, self.fixed_values['theta']]
            length = self.H_params['L'][L]
            energy_density = energy/(length-1)
            plt.plot(energy_density, label="L = {}".format(length))

        x_dist = 10
        plt.xticks(np.arange(0, energies[0,0].shape[0], x_dist))
        plt.xlabel("Excited state n")
        plt.ylabel(r"Energy per site ${E_n/L}$")

        plt.savefig(self.simulation_path + 'figures/spectrum')

        plt.legend()
        plt.show()


    def plot_fitted_gaps(self, gaps, alpha=-1, n_gap=0):
        """
        Plots the energy gaps as a function of L**(alpha), usually alpha=-1
        """

        sns.set_style('whitegrid')
        fig = plt.figure()
        fig.suptitle("Convergence of energy gaps as a function of L")

        for i in range(len(self.H_params['theta'])):
            x = [float(L) ** alpha for L in self.H_params['L']]
            y = gaps[n_gap,:,i]
            fit = np.polyfit(x, y, 1)

            p = np.poly1d(fit)
            xp = np.linspace(0, x[0], 100)

            pl = plt.plot(x, y, '.')
            ratio = (self.H_params['theta'][i] / np.pi).as_integer_ratio()
            if ratio[0] == 0:
                plt.plot(xp, p(xp), '-', label=r"0", c=pl[0].get_c())
            else:
                plt.plot(xp, p(xp), '-', label=r"$\frac{{{}}}{{{}}}$".format(ratio[0], ratio[1]), c=pl[0].get_c())

        plt.xlabel(r"$L^{{{}}}$".format(alpha))
        plt.ylabel(r"$\Delta$")
        plt.legend(title=r"$\frac{\theta}{\pi}$:")

        plt.savefig(self.simulation_path + 'figures/fitted_gaps')
        plt.show()


    # def plot_scalings(self, observable, critical_exponent, constant, rescale=False):
    #     """
    #     Plots the scaling of an observable at the critical point and fits a critical exponent.
    #     """
    #     #TODO
    #     sns.set_style('whitegrid')

    #     for i in  range(len(self.H_params["theta"])):
    #         theta = self.H_params["theta"][i]
            
    #         z = -round(critical_exponent, 2)
    #         scalar = 0 if rescale == False else constant

    #         x = [np.log(L) for L in self.H_params['L']]
    #         y = [np.log(gap) - scalar for gap in observable[:,i]]
            
    #         p = np.poly1d([critical_exponent, constant - scalar])
    #         x0 = min(x[-1], x[0]) if rescale == False else 0
    #         x_fit = np.linspace(x0, max(x[-1], x[0]), 100)
    #         y_fit = p(x_fit)

    #         ratio = (theta / np.pi).as_integer_ratio()
    #         if ratio[0] == 0:
    #             label = r"$\theta$: 0,      " +  r"$z$: {}".format(z)
    #         else:
    #             label = r"$\theta$: " +  r"$\frac{{{}}}{{{}}} \pi$,      $z$: {}".format(ratio[0], ratio[1], z)
    #         pl = plt.plot(x, y, '.')
    #         plt.plot(x_fit, y_fit, '-', label=label, c=pl[0].get_c())

    #     plt.xlabel(r"$\log L$")
    #     plt.ylabel(r"$\log \Delta$")
    #     plt.legend()

        # if rescale == True:
        #     plt.title(r"Fit of the scaling behaviour for the gaps (setting constant to zero)")
        #     path_figure = self.simulation_path + 'figures/scaling_gaps_scaled'
        # else:
        #     plt.title(r"Fit of the scaling behaviour for the gaps")
        #     path_figure = self.simulation_path + 'figures/scaling_gaps'
        # plt.savefig(path_figure)
        # plt.show()

    def plot_gap_scalings(self, gaps, H_params, fixed_values, rescale=False, n_gap=0):
        #TODO

        sns.set_style('whitegrid')
        gaps = gaps[n_gap,:,:]
        for i in range(len(H_params["theta"])):
            theta = H_params["theta"][i]
            
            critical_exponent, constant = t.compute_critical_exponent(gaps, H_params, fixed_values, theta)
            z = -round(critical_exponent, 2)
            scalar = 0 if rescale == False else constant

            x = [np.log(L) for L in H_params['L']]
            y = [np.log(gap) - scalar for gap in gaps[:,i]]
            
            p = np.poly1d([critical_exponent, constant - scalar])
            x0 = min(x[-1], x[0]) if rescale == False else 0
            x_fit = np.linspace(x0, max(x[-1], x[0]), 100)
            y_fit = p(x_fit)

            ratio = (theta / np.pi).as_integer_ratio()
            if ratio[0] == 0:
                label = r"$\theta$: 0,      " +  r"$z$: {}".format(z)
            else:
                label = r"$\theta$: " +  r"$\frac{{{}}}{{{}}} \pi$,      $z$: {}".format(ratio[0], ratio[1], z)
            pl = plt.plot(x, y, '.')
            plt.plot(x_fit, y_fit, '-', label=label, c=pl[0].get_c())

        plt.xlabel(r"$\log L$")
        plt.ylabel(r"$\log \Delta$")
        plt.legend()

        if rescale == True:
            plt.title(r"Fit of the scaling behaviour for the gaps (setting constant to zero)")
            path_figure = self.simulation_path + 'figures/scaling_gaps_scaled'
        else:
            plt.title(r"Fit of the scaling behaviour for the gaps")
            path_figure = self.simulation_path + 'figures/scaling_gaps'
        plt.savefig(path_figure)
        plt.show()
        
        

    def plot_correlation_scalings(self, correlations, H_params, fixed_values, rescale=False, name_suffix=""):
        #TODO
        sns.set_style('whitegrid')

        for i in  range(len(H_params["theta"])):
            theta = H_params["theta"][i]
            
            critical_exponent, constant = t.compute_critical_exponent(correlations, H_params, fixed_values, theta)
            eta = round(2 - critical_exponent, 2)
            scalar = 0 if rescale == False else constant

            x = [np.log(L) for L in H_params['L']]
            y = [np.log(corr) - scalar for corr in correlations[:,i]]
            
            p = np.poly1d([critical_exponent, constant - scalar])
            x0 = min(x[-1], x[0]) if rescale == False else 0
            x_fit = np.linspace(x0, max(x[-1], x[0]), 100)
            y_fit = p(x_fit)

            ratio = (theta / np.pi).as_integer_ratio()
            if ratio[0] == 0:
                label = r"$\theta$: 0,      " +  r"$\eta$: {}".format(eta)
            else:
                label = r"$\theta$: " +  r"$\frac{{{}}}{{{}}} \pi$,      $\eta$: {}".format(ratio[0], ratio[1], eta)
            pl = plt.plot(x, y, '.')
            plt.plot(x_fit, y_fit, '-', label=label, c=pl[0].get_c())

        plt.xlabel(r"$\log L$")
        plt.ylabel(r"$\log S$")
        plt.legend()

        if rescale == True:
            plt.title(r"Fit of the scaling behaviour for the correlations (setting constant to zero)")
            path_figure = self.simulation_path + 'figures/scaling_corr_scaled'
        else: 
            plt.title(r"Fit of the scaling behaviour for the correlations")
            path_figure = self.simulation_path + 'figures/scaling_corr'

        plt.savefig(path_figure)
        plt.show()


    def plot_central_charges_polar(self, charges, H_params, charge_type='fit', name_suffix=""):
        """
        Plots a circular topography diagram for a given observable.
        """

        sns.set_style('whitegrid')
        _, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        j = 0 if charge_type == 'fit' else -1
        for i in range(len(H_params["L"]) + j):
            ax.plot(H_params["theta"], charges[i], '-1', label=H_params["L"][i])

        ax.set_rmax(4)
        ax.set_rticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])  # Less radial ticks
        ax.set_rlabel_position(-10)  # Move radial labels away from plotted line
        ax.legend(title="L:")
        ax.grid(True)

        ax.set_title("Central charges for the interpolating hamiltonian")

        plt.savefig(self.simulation_path + 'figures/central_charges_polar' + name_suffix)
        plt.show()


    def plot_scalings(self, scalings, name_suffix=''):
        """
        Plots scaling dimensions coming from MERA
        """
        sns.set_style('whitegrid')
        for i in range(len(self.H_params['theta'])):
            theta = self.H_params['theta'][i]
            ratio = (theta/np.pi).as_integer_ratio()
            label = r"$\frac{{{}}}{{{}}}$".format(ratio[0], ratio[1]) if ratio[0] != 0 else "0"
            plt.plot(range(1,11), scalings[0][i], '--p', linewidth=0.5, label=label)
        plt.title('Sutherland scaling dimensions')
        plt.xlabel('k')
        plt.ylabel(r'Scaling Dims: $\Delta_k$')
        plt.legend(title=r'$\theta / \pi$')
        plt.savefig(self.simulation_path + 'figures/scaling_dimensions' + name_suffix)
        plt.show()


    # def plot_scaling_dimensions(self, scalings):
    #     sns.set_style('whitegrid')
    #     plt.plot(self.H_params['theta'], np.stack(scalings[0]), 'rx')
    #     plt.show()