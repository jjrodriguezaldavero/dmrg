
import numpy as np

from tools import tools
from tools.Plotter.PlotterANNNP import PlotterANNNP

# Hamiltonian parameters for the point C1: (U, F) = (0, 1) performing a cut along the F parameter
H_params = {
    'L': np.array([64, 70, 76, 82, 88, 94, 100]), 
    'F': tools.compute_fss_range(central_value=1, n=8, delta=0.004),
    'U': np.array([0]),
    'V': np.array([-0.2, 0.0, 0.2])
}

def run(workers, simulation_path, parallel, use_cluster):

    """Plot Phase Diagram"""
    Plotter = PlotterANNNP(H_params, simulation_path)
    Plotter.plot_ANNNP_phase_diagram()