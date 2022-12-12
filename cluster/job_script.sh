#!/bin/bash

#$ -q $QUEUENAME
#$ -cwd
#$ -N $NAME
#$ -m bea
#$ -M juanjolh97@gmail.com
#$ -b no
#$ -j yes

#$ -pe mpich $NUMCORES

echo STARTED on $(date)

# Limit MKL parallelization because I am actually using multiprocessing
export MKL_NUM_THREADS=$WORKERCORES
export OMP_NUM_THREADS=$WORKERCORES

# Job
python3 /nethome/6835384/dmrg/$SIMULATION

# Resource consumption
qstat -j $JOB_ID | awk 'NR==1,/^scheduling info:/'

echo FINISHED on $(date)