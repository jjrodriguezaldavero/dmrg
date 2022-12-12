
# Default core configuration found in config.json
for keyval in $(grep -E '": [^\{]' config.json | sed -e 's/: /=/' -e "s/\(\,\)$//"); do
    eval export $keyval
done

# Other option: choose individually by
# export NUMCORES=24 
# export WORKERCORES=1
# export WORKERS=24

#Submission queue:

###############################################
# export NUMCORES=24 
# export WORKERCORES=1
# export WORKERS=24

# sed -e 's/$NUMCORES/'$NUMCORES'/g' \
#     -e 's/$WORKERCORES/'$WORKERCORES'/g' \
#     -e 's/$QUEUENAME/'$QUEUENAME'/g' \
#     -e 's/$NAME/'FF_G2'/g' \
#     -e 's/$SIMULATION/'FF_G2.py'/g' job_script.sh > temp.sh
# qsub -V temp.sh

# sed -e 's/$NUMCORES/'$NUMCORES'/g' \
#     -e 's/$WORKERCORES/'$WORKERCORES'/g' \
#     -e 's/$QUEUENAME/'$QUEUENAME'/g' \
#     -e 's/$NAME/'FF_G3'/g' \
#     -e 's/$SIMULATION/'FF_G3.py'/g' job_script.sh > temp.sh
# qsub -V temp.sh

# sed -e 's/$NUMCORES/'$NUMCORES'/g' \
#     -e 's/$WORKERCORES/'$WORKERCORES'/g' \
#     -e 's/$QUEUENAME/'$QUEUENAME'/g' \
#     -e 's/$NAME/'FSS_G2'/g' \
#     -e 's/$SIMULATION/'FSS_G2.py'/g' job_script.sh > temp.sh
# qsub -V temp.sh

# sed -e 's/$NUMCORES/'$NUMCORES'/g' \
#     -e 's/$WORKERCORES/'$WORKERCORES'/g' \
#     -e 's/$QUEUENAME/'$QUEUENAME'/g' \
#     -e 's/$NAME/'RT_SU_C'/g' \
#     -e 's/$SIMULATION/'RT_SU_C.py'/g' job_script.sh > temp.sh
# qsub -V temp.sh

# sed -e 's/$NUMCORES/'$NUMCORES'/g' \
#     -e 's/$WORKERCORES/'$WORKERCORES'/g' \
#     -e 's/$QUEUENAME/'$QUEUENAME'/g' \
#     -e 's/$NAME/'RT_SU_S'/g' \
#     -e 's/$SIMULATION/'RT_SU_S.py'/g' job_script.sh > temp.sh
# qsub -V temp.sh


# export NUMCORES=32
# export WORKERCORES=1
# export WORKERS=32
# sed -e 's/$NUMCORES/'$NUMCORES'/g' \
#     -e 's/$WORKERCORES/'$WORKERCORES'/g' \
#     -e 's/$QUEUENAME/'$QUEUENAME'/g' \
#     -e 's/$NAME/'RT_FU'/g' \
#     -e 's/$SIMULATION/'RT_FU.py'/g' job_script.sh > temp.sh
# qsub -V temp.sh
#########################################################################


# # ######################### MERA ##############################
export NUMCORES=1
export WORKERCORES=1
export WORKERS=1

sed -e 's/$NUMCORES/'$NUMCORES'/g' \
    -e 's/$WORKERCORES/'$WORKERCORES'/g' \
    -e 's/$QUEUENAME/'$QUEUENAME'/g' \
    -e 's/$NAME/'MERA_C1'/g' \
    -e 's/$SIMULATION/'MERA_C1.py'/g' job_script.sh > temp.sh
qsub -V temp.sh

sed -e 's/$NUMCORES/'$NUMCORES'/g' \
    -e 's/$WORKERCORES/'$WORKERCORES'/g' \
    -e 's/$QUEUENAME/'$QUEUENAME'/g' \
    -e 's/$NAME/'MERA_G2'/g' \
    -e 's/$SIMULATION/'MERA_G2.py'/g' job_script.sh > temp.sh
qsub -V temp.sh

export NUMCORES=13
export WORKERCORES=1
export WORKERS=13

sed -e 's/$NUMCORES/'$NUMCORES'/g' \
    -e 's/$WORKERCORES/'$WORKERCORES'/g' \
    -e 's/$QUEUENAME/'$QUEUENAME'/g' \
    -e 's/$NAME/'MERA_SU'/g' \
    -e 's/$SIMULATION/'MERA_SU.py'/g' job_script.sh > temp.sh
qsub -V temp.sh
# #########################################################################

rm -r temp.sh