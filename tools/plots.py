"""
Library with plotting routines.
"""

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from tools import tools as t


def plot_central_charges(charges, ranges, simulation_path, fixed_values=None):
    """
    Plots the central charges with respect to two fixed values.
    """    
    # Crops the multidimensional array to consider the two fixed values.
    cropped_array, range_names, range_values = t.crop_array(charges, ranges, fixed_values)
    
    # Plots the cropped array using Seaborn
    sns.set_style('whitegrid')
    plt.plot(cropped_array, '-1')
    plt.legend(range_values[1], title=range_names[1])

    x_tick_periodicity = 2
    plt.xticks(np.arange(0, len(range_values[0]), x_tick_periodicity), np.around(range_values[0], 4)[0::x_tick_periodicity])

    title = r'Finite-size scaling for the central charge $c(L)$'
    plt.title(title)
    
    ylabel = 'Central charge c'
    plt.xlabel(r"$\{}$".format(range_names[0]))
    plt.ylabel(ylabel)

    path_figure = simulation_path + 'figures/central_charges'
    plt.savefig(path_figure)
    plt.show()


def plot_finite_size(observable, ranges, fixed_values, critical_exponent, title, ylabel, path_figure):
    """
    Auxiliar function to plot observables that show finite size effects with a critical exponent.
    """
    # Crops the multidimensional array to consider the two fixed values.
    cropped_array, range_names, range_values = t.crop_array(observable, ranges, fixed_values)

    # Plots the cropped array using Seaborn
    sns.set_style('whitegrid')
    plt.plot(cropped_array * range_values[1] ** critical_exponent, '-1')
    plt.legend(range_values[1], title=range_names[1])

    x_tick_periodicity = 2
    plt.xticks(np.arange(0, len(range_values[0]), x_tick_periodicity), np.around(range_values[0], 4)[0::x_tick_periodicity])
    
    plt.title(title)
    plt.xlabel(r"$\{}$".format(range_names[0]))
    plt.ylabel(ylabel)

    plt.savefig(path_figure)
    plt.show()


def plot_finite_size_gaps(gaps, ranges, fixed_values, crit_x=None, n_gap=0):
    """
    Plots the finite size effects for the energy gaps. The critical x-value must be specified in advance.
    """
    # Computes the critical exponent by fitting the energy gaps at the critical point.
    if crit_x is not None:
        critical_exponent, _ = t.compute_critical_exponent(gaps[n_gap], ranges, fixed_values, crit_x)
        z = round(-critical_exponent, 3)
    else:
        critical_exponent = z = 0

    # Plots the energy gaps using the auxiliar function.
    title = r'Finite-size scaling for the energy gap $\Delta$ for $z = {}$'.format(z)
    ylabel = r'$L^{{{}}}\cdot\Delta$'.format(round(-critical_exponent, 3))
    path_figure = 'figures/finite_size_gaps'
    plot_finite_size(gaps[n_gap], ranges, fixed_values, -critical_exponent, title, ylabel, path_figure)


def plot_finite_size_betas(betas, ranges, fixed_values, crit_x=None):
    """
    Plots the finite size effects for the Callan-Symanzik beta functions. The critical x-value must be specified in advance.
    """
    # Computes the critical exponent by fitting the energy gaps at the critical point.
    if crit_x is not None:
        critical_exponent, _ = t.compute_critical_exponent(abs(betas), ranges, fixed_values, crit_x)
        nu = -critical_exponent
    else:
        critical_exponent = nu = 0

    # Plots the Callan-Symanzik betas using the auxiliar function.
    title = r'Finite-size scaling for the Callan-Symanzik $\beta$ function for $\nu = {}$'.format(round(nu, 3))
    ylabel = r'$L^{{{}}}\cdot\beta$'.format(round(-critical_exponent, 3))
    path_figure = 'figures/finite_size_betas'
    plot_finite_size(betas, ranges, fixed_values, -critical_exponent, title, ylabel, path_figure)


def plot_finite_size_correlations(correlations, ranges, fixed_values, crit_x=None):
    """
    Plots the finite size effects for the spin-spin correlation functions. The critical x-value must be specified in advance.
    """
    # Computes the critical exponent by fitting the energy gaps at the critical point.
    if crit_x is not None:
        critical_exponent, _ = t.compute_critical_exponent(correlations, ranges, fixed_values, crit_x)
        eta = round(2 - critical_exponent, 3)
    else:
        critical_exponent = eta = 0

    # Plots the Callan-Symanzik betas using the auxiliar function.
    title = r'Finite-size scaling for the two-point correlator $S(L)$ with $\eta = {}$'.format(eta)
    ylabel = r'$L^{{{}}}\cdot S(L)$'.format(round(-critical_exponent, 3))
    path_figure = 'figures/finite_size_correlations'
    plot_finite_size(correlations, ranges, fixed_values, -critical_exponent, title, ylabel, path_figure, )


def plot_topography(charges, ranges, fixed_values, cmap="jet", title="Observable", path_figure="topography"):
    """
    Plots a 2D rough topography diagram for a given observable.
    """
    # Crops the multidimensional array to consider the two fixed values.
    cropped_array, range_names, range_values = t.crop_array(charges, ranges, fixed_values)

    # Plots the cropped_array using Matplotlib's heatmap
    plt.imshow(np.flip(cropped_array.T, axis=0), cmap=cmap)

    x_tick_periodicity = 10
    plt.xticks(np.arange(0, len(range_values[0]-1), x_tick_periodicity), np.around(range_values[0], 4)[0::x_tick_periodicity])
    y_tick_periodicity = 10
    plt.yticks(np.arange(0, len(range_values[1]-1), y_tick_periodicity), np.flip(np.around(range_values[1], 4))[0::y_tick_periodicity])

    # title = 'Central charge'
    plt.title(title)
    plt.colorbar()
    plt.xlabel(r"${}$".format(range_names[0]))
    plt.ylabel(r"${}$".format(range_names[1]))

    # path_figure = 'figures/charge_topography'
    plt.savefig('figures/' + path_figure)
    plt.show()


def plot_critical_plane(points, surf_type="linear"):
    """
    Plots a linear or quadratic function that better fits a set of points.
    Implemented only for the ANNNP model.
    """

    # Extracts the x, y and z values from the set of points
    F_data = [point[0] for point in points] # F
    U_data = [point[1] for point in points] # U
    V_data = [point[2] for point in points] # V

    # Computes a 2D mesh from the x and y values of the data (needed for the plot, not for the fit).
    U, V = np.meshgrid(np.linspace(min(U_data), max(U_data), 21), 
                       np.linspace(min(V_data), max(V_data), 21))

    # Define fitting functions
    def linear_surface(data, a, b, c):
        """Analytical formula of plane for the fitting."""
        x = data[0]
        y = data[1]
        return a + b*x + c*y

    def quadratic_surface(data, a, b, c, d, e):
        """Analytical formula of a quadratic surface for the fitting."""
        x = data[0]
        y = data[1]
        return a + b*x + c*y + d*x**2 + e*y**2

    # Fit the parameters of the function to the points and evaluate the surface to the parameters.
    if surf_type == 'linear':
        parameters, _ = curve_fit(linear_surface, [U_data, V_data], F_data)
        F = linear_surface(np.array([U, V]), *parameters)
        equation = "f(U,V) = {} + {}U + {}V".format(
            round(parameters[0], 3), round(parameters[1], 3), round(parameters[2], 3))
    elif surf_type == 'quadratic':
        parameters, _ = curve_fit(quadratic_surface, [U_data, V_data], F_data)
        F = quadratic_surface(np.array([U, V]), *parameters)
        equation = "f(U,V) = {} + {}U + {}V + {}U^2 + {}V^2".format(
            round(parameters[0], 3), round(parameters[1], 3), round(parameters[2], 3), round(parameters[3], 3), round(parameters[4], 3))

    # Plot the result
    fig = plt.figure()
    fig.suptitle(equation)

    ax = Axes3D(fig, computed_zorder=False)
    ax.plot_surface(U, V, F, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)
    ax.scatter(U_data, V_data, F_data, color='black', marker='o')

    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.set_zlabel("f(U,V)")

    ticks = 5
    ax.xaxis.set_ticks(np.linspace(min(U_data), max(U_data), ticks))
    ax.yaxis.set_ticks(np.linspace(min(V_data), max(V_data), ticks))

    path_figure = 'figures/critical_plane'
    plt.savefig(path_figure)
    plt.show()


def plot_spectrum(energies, ranges, fixed_values):
    fig = plt.figure()
    fig.suptitle("Energy spectrum")

    for L in range(energies.shape[0]):
        energy = energies[L,fixed_values['theta']]
        length = ranges['L'][L]
        energy_density = energy/(length-1)
        plt.plot(energy_density, label="L = {}".format(length))

    x_dist = 10
    plt.xticks(np.arange(0, energies[0,0].shape[0], x_dist))
    plt.xlabel("Excited state n")
    plt.ylabel(r"Energy per site ${E_n/L}$")

    plt.legend()
    plt.show()


def plot_fitted_gaps(gaps, ranges, alpha=-1, n_gap=0):
    sns.set_style('whitegrid')
    fig = plt.figure()
    fig.suptitle("Convergence of energy gaps as a function of L")

    for i in range(len(ranges['theta'])):
        x = [float(L) ** alpha for L in ranges['L']]
        y = gaps[n_gap,:,i]
        fit = np.polyfit(x, y, 1)

        p = np.poly1d(fit)
        xp = np.linspace(0, x[0], 100)

        pl = plt.plot(x, y, '.')
        ratio = (ranges['theta'][i] / np.pi).as_integer_ratio()
        if ratio[0] == 0:
            plt.plot(xp, p(xp), '-', label=r"0", c=pl[0].get_c())
        else:
            plt.plot(xp, p(xp), '-', label=r"$\frac{{{}}}{{{}}}$".format(ratio[0], ratio[1]), c=pl[0].get_c())

    plt.xlabel(r"$L^{{{}}}$".format(alpha))
    plt.ylabel(r"$\Delta$")
    plt.legend(title=r"$\frac{\theta}{\pi}$:")

    path_figure = 'figures/fitted_gaps'
    plt.savefig(path_figure)
    plt.show()


def plot_gap_scalings(gaps, ranges, fixed_values, rescale=False):

    sns.set_style('whitegrid')

    for i in  range(len(ranges["theta"])):
        theta = ranges["theta"][i]
        
        critical_exponent, constant = t.compute_critical_exponent(gaps, ranges, fixed_values, theta)
        z = -round(critical_exponent, 2)
        scalar = 0 if rescale == False else constant

        x = [np.log(L) for L in ranges['L']]
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
        path_figure = 'figures/scaling_gaps_scaled'
    else:
        plt.title(r"Fit of the scaling behaviour for the gaps")
        path_figure = 'figures/scaling_gaps'
    plt.savefig(path_figure)
    plt.show()
    
    

def plot_correlation_scalings(correlations, ranges, fixed_values, rescale=False):

    sns.set_style('whitegrid')

    for i in  range(len(ranges["theta"])):
        theta = ranges["theta"][i]
        
        critical_exponent, constant = t.compute_critical_exponent(correlations, ranges, fixed_values, theta)
        eta = round(2 - critical_exponent, 2)
        scalar = 0 if rescale == False else constant

        x = [np.log(L) for L in ranges['L']]
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
        path_figure = 'figures/scaling_corr_scaled'
    else: 
        plt.title(r"Fit of the scaling behaviour for the correlations")
        path_figure = 'figures/scaling_corr'

    plt.savefig(path_figure)
    plt.show()