import sys
import os


CWD = os.getcwd()
SIMULATION_PATH = os.path.dirname(os.path.realpath(__file__))

###### UPSTREAM OR DOWNSTREAM ######
SYNC_DIRECTION = "UPSTREAM DELETE"
################################

sys.path.append(CWD)
from tools.tools import synchronize

if __name__ == "__main__":
    synchronize(CWD, SIMULATION_PATH, SYNC_DIRECTION)


# import pickle
# with open('simulations/Sutherland/ScalingDimensions/Segment/data/checkpoints/CP_L2.0_theta0.0', "rb") as f:
#         test = pickle.load(f)