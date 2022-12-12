"""
Library with plotting routines specific for the ANNNP model.
"""

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from tools.Plotter.Plotter import Plotter
from tools import tools as t


class PlotterANNNP(Plotter):

    def plot_topography(self, charges, fixed_values, title="Rough topography", cmap="jet", name_suffix=''):
        """
        Plots a 2D rough topography diagram for a given observable.
        """
        # Crops the multidimensional array to consider the two fixed values.
        cropped_array, range_names, range_values = t.crop_array(charges, self.H_params, fixed_values)

        # Plots the cropped_array using Matplotlib's heatmap
        plt.imshow(np.flip(cropped_array.T, axis=0), cmap=cmap)

        x_tick_periodicity = 10
        plt.xticks(np.arange(0, len(range_values[0]), x_tick_periodicity), np.around(range_values[0], 0)[0::x_tick_periodicity])
        y_tick_periodicity = 10
        plt.yticks(np.arange(0, len(range_values[1]-1), y_tick_periodicity), np.flip(np.around(range_values[1], 4))[0::y_tick_periodicity])

        plt.title(title)
        plt.colorbar()
        plt.xlabel(r"${}$".format(range_names[0]))
        plt.ylabel(r"${}$".format(range_names[1]))

        plt.savefig(self.simulation_path + 'figures/' + 'topography' + name_suffix)
        plt.show()


    def plot_critical_plane(self, points, surf_type="linear"):
        """
        Plots a linear or quadratic function that better fits a set of points.
        Needs preprocessing, its input is a list of critical points [p1, p2, p3...] where each point
        is a tuple of the form (F_val, U_val, V_val).
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

        plt.savefig(self.simulation_path + 'figures/critical_plane')
        plt.show()


    def plot_ANNNP_phase_diagram(self, peschel=True, plane=True):
        """
        Plot the phase diagram together with the critical plane and the Peschel-Emery line
        """

        # Set axes and plot dimensions
        ax = plt.figure().add_subplot(projection='3d')
        ax.grid(False)
        U0, U1 = -1.5, 1.5
        F0, F1 = -1, 4
        V0, V1 = -0.2, 0.2
        ax.set_xlim([U0, U1])
        ax.set_ylim([V0, V1])
        ax.set_zlim([F0, F1])
        
        # Highlight the V=0 plane for the trivial and topological phases
        U = np.linspace(U0, U1, 100)
        line1 = np.poly1d(np.polyfit([-1, 0, 1],[0, 1, 1+np.sqrt(3)], 2))
        critical_line = np.piecewise(U, [U < -1, U >= -1], [0, lambda x: line1(x)])
        line2 = np.poly1d(np.polyfit([-1, -0.75, -0.5],[0, -0.4, F0], 2))
        XXZ_line = np.piecewise(U, [U < -1, U >= -1, U >= -0.5], [0, lambda x: line2(x), F0])
        

        ax.add_collection3d(plt.fill_between(U, critical_line, F0, color='y', alpha=0.2, zorder=1), zs=0, zdir='y')
        ax.text(-1, 0, 1.5, 'Paramagnetic \n (trivial)', size=10, zorder=20, color='k')
        ax.add_collection3d(plt.fill_between(U, critical_line, F1, color='r', alpha=0.2, zorder=1), zs=0, zdir='y')
        ax.text(0.6, 0, 0.5, r'$\mathbb{Z}_3$'+'-ordered \n (topological)', size=10, zorder=20, color='k')
        ax.add_collection3d(plt.fill_between(U, XXZ_line, -1, color='b', alpha=0.2, zorder=1), zs=0, zdir='y')
        ax.text(-1.4, 0, -0.5, 'Exotic phases', size=10, zorder=20, color='k')

        # Highlight horizontal and vertical lines
        ax.plot([U0, U1], [0,0], [0,0], color='k', linewidth=0.5, zorder=8)
        ax.plot([0,0], [0,0], [F0, F1], color='k', linewidth=0.5, zorder=8)

        # Compute and plot the Peschel-Emery line
        if peschel == True:
            r_arr = np.linspace(0.3, 1.8, 101)
            U_pe = [t.compute_peschel_emery(r)['U'] for r in r_arr]
            F_pe = [t.compute_peschel_emery(r)['F'] for r in r_arr]
            V_pe = [t.compute_peschel_emery(r)['V'] for r in r_arr]
            ax.plot(U_pe, V_pe, F_pe, label="Peschel-Emery line", zorder=20, linewidth=1.5)
            ax.plot(U_pe, V_pe, [0 for u in range(len(U_pe))], label="XY projection", zorder=20, linewidth=0.5, color='b', linestyle='--', alpha=0.5)

        # Plot critical plane
        if plane == True:
            UU, VV = np.meshgrid(np.linspace(0, 1.5, 10), np.linspace(-0.08, 0.08, 10))
            F_crit = lambda u, v: 0.915 + 1.865 * u + 1.662 * v
            c = ax.plot_surface(UU, VV, F_crit(UU, VV), label="Potts transition critical plane", alpha=0.5, color='dimgray', zorder=5)
            #bug
            c._facecolors2d=c._facecolor3d
            c._edgecolors2d=c._edgecolor3d

        # Highlight G1
        ax.scatter(0, 0, 0, color='b')
        ax.text(0, 0, 0, 'G1', size=10, zorder=10, color='k')

        # Highlight G2
        ax.scatter(1, 0, 1 + np.sqrt(3), color='b')
        ax.text(1, 0, 1 + np.sqrt(3), 'G2', size=10, zorder=10, color='k')

        # Plot critical line between C1, C2 and G2
        plt.plot(U, 0*U, critical_line, color='k', linewidth=1.2)

        # Highlight C1
        ax.scatter(0, 0, 1, color='r')
        ax.text(0, 0, 1, 'C1', size=10, zorder=10, color='k')

        # Highlight C2
        ax.scatter(-1, 0, 0, color='r')
        ax.text(-1, 0, 0, 'C2', size=10, zorder=10, color='k')

        # Set labels, ticks and legend
        ax.set_xlabel("U")
        ax.set_ylabel("V")
        ax.set_zlabel("f")

        ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.legend()

        plt.savefig(self.simulation_path + 'figures/ANNNP_phase_diagram')
        plt.show()


    def plot_gap_convergence(self, gaps, n_gap=0):
        """
        Fits an exponential function to the convergence of the energy gaps at a point.
        """
        #TODO
        # Selects the data
        L = self.H_params['L']
        gaps = gaps[n_gap, :, 0, 0, 0]

        fit = np.polyfit(L, np.log(gaps), 1)    
        a = np.exp(fit[1])
        b = fit[0]

        sns.set_style('whitegrid')
        plt.plot(L, gaps, '.', label="DMRG")
        plt.plot(L, a * np.exp(L*b), '-', label="Exponential fit")
        plt.title(r"Log energy gaps at G2 with $\Delta(L) \approx {} \cdot e^{{{} \cdot L}}$".format(round(a,2), round(b,2)))
        plt.xlabel(r"$L$")
        plt.ylabel(r"$\Delta(L)$")
        plt.yscale('log')
        
        plt.legend()

        path_figure = 'figures/gaps_convergence'
        plt.savefig(path_figure)
        plt.show()


    def plot_frustration_free_fit(self, gaps, fixed_values, name_suffix=''):
        """
        Plots the Frustration-Free effects of the energy gaps close to the Peschel-Emery line.
        """
        # Crops the multidimensional array to consider the two fixed values.
        cropped_array, range_names, range_values = t.crop_array(gaps[0], self.H_params, fixed_values)

        def exponential_fit(L, a, b):
            return a * np.exp(-b * L)

        
        x = range_values[1]
        y = cropped_array[-1]

        # popt, _ = curve_fit(exponential_fit, x, y)
        sns.set_style('whitegrid')
        plt.plot(x,y, '-1')
        plt.xlabel(r"$L$")
        plt.ylabel(r"$\Delta(L)$")
        # eqn =  r"$\Delta(L) = %s \cdot e^{-%s L}$" % (round(popt[0],2), round(popt[1],2))
        # plt.title("Frustration free efects near the Peschel-Emery line: " + eqn)
           
        plt.show()
        #print(popt)
        # Build mesh
    #     X, Y = np.meshgrid(range_values[0], range_values[1])

    #     # Define fitting function
    #     def exponential_fit(p, L, alpha, beta):
    #         return  p ** alpha * np.exp(-beta * L)

    #     Z = cropped_array.T

    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.plot_surface(X, Y, Z, cmap='plasma')
    #     ax.set_zlim(0,np.max(Z))
    #     plt.show()

    #     # Callable passed to curve_fit
    #     def _exponential_fit(M, alpha, beta):
    #         x, y = M
    #         return exponential_fit(x, y, alpha, beta)

    #     xdata = np.vstack((X.ravel(), Y.ravel()))
    #     popt, _ = curve_fit(_exponential_fit, xdata, Z.ravel())
    #     fit = exponential_fit(X, Y, *popt)
    #     print("Fitted parameters:")
    #     print(popt)

    #     # Plot the 3D figure of the fitted function and the residuals.
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(projection='3d')
    #     # ax.plot_surface(X, Y, fit, cmap='plasma')
    #     # cset = ax.contourf(X, Y, Z-fit, zdir='z', offset=-4, cmap='plasma')
    #     # ax.set_zlim(0,np.max(fit))
    #     # plt.show()
    
    def plot_frustration_free_planes(self, gaps):
        gaps = gaps[0]
        x = self.H_params['F']
        y = self.H_params['U']
        L = self.H_params['L']
        X, Y = np.meshgrid(x, y)

        def log_tick_formatter(val, pos=None):
            return r"$10^{{{:.0f}}}$".format(val)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        import matplotlib.ticker as mticker
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        for i in range(gaps.shape[0]):
            Z = gaps[i,:,:,0]
            Z[Z != 0] = np.log10(Z[Z != 0])
            Z[Z == 0] = None
            ax.plot_surface(X,Y,Z, label=L[i])

        plt.legend(title='L:')
        plt.show()