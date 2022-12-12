import sys
import os


CWD = os.getcwd()
SIMULATION_PATH = os.path.dirname(os.path.realpath(__file__))

###### UPSTREAM OR DOWNSTREAM ######
SYNC_DIRECTION = "DOWNSTREAM"
################################

sys.path.append(CWD)
from tools.tools import synchronize

if __name__ == "__main__":
    synchronize(CWD, SIMULATION_PATH, SYNC_DIRECTION)
