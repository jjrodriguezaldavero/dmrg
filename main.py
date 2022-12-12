
import sys
import json
import logging


###########################################################
SIMULATION_PATH = "simulations/Sutherland/RoughTopography/Segment/" # IMPORTANT: it must end in /
###########################################################
USE_CLUSTER = False
PARALLEL = False
VERBOSE = False
###########################################################


if VERBOSE: logging.basicConfig(level=logging.INFO)

if USE_CLUSTER:
    CLUSTER_PATH = "/nethome/6835384/dmrg/"
    SIMULATION_PATH = CLUSTER_PATH + SIMULATION_PATH
    with open(CLUSTER_PATH + "cluster/config.json", "r") as f:
        WORKERS = json.load(f)["WORKERS"]
else:
    WORKERS = 4

sys.path.append(SIMULATION_PATH)

from simulation import run


if __name__ == "__main__":
    run(WORKERS, SIMULATION_PATH, PARALLEL, USE_CLUSTER)