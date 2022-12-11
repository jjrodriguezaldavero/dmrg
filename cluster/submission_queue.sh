
# Default core configuration found in config.json
for keyval in $(grep -E '": [^\{]' config.json | sed -e 's/: /=/' -e "s/\(\,\)$//"); do
    eval export $keyval
done

# Other option: choose individually by
# export NUMCORES=24 
# export WORKERCORES=1
# export WORKERS=24

#Submission queue:
# ############################ Short and light load ########################
export NUMCORES=9
export WORKERCORES=1
export WORKERS=9

sed -e 's/$NUMCORES/'$NUMCORES'/g' \
    -e 's/$WORKERCORES/'$WORKERCORES'/g' \
    -e 's/$QUEUENAME/'$QUEUENAME'/g' \
    -e 's/$NAME/'SU_SD'/g' \
    -e 's/$SIMULATION/'SU_SD.py'/g' job_script.sh > temp.sh
qsub -V temp.sh

export NUMCORES=1
export WORKERCORES=1
export WORKERS=1

sed -e 's/$NUMCORES/'$NUMCORES'/g' \
    -e 's/$WORKERCORES/'$WORKERCORES'/g' \
    -e 's/$QUEUENAME/'$QUEUENAME'/g' \
    -e 's/$NAME/'C1_SD'/g' \
    -e 's/$SIMULATION/'ANNNP_C1_SD.py'/g' job_script.sh > temp.sh
qsub -V temp.sh

sed -e 's/$NUMCORES/'$NUMCORES'/g' \
    -e 's/$WORKERCORES/'$WORKERCORES'/g' \
    -e 's/$QUEUENAME/'$QUEUENAME'/g' \
    -e 's/$NAME/'C2_SD'/g' \
    -e 's/$SIMULATION/'ANNNP_C2_SD.py'/g' job_script.sh > temp.sh
qsub -V temp.sh

sed -e 's/$NUMCORES/'$NUMCORES'/g' \
    -e 's/$WORKERCORES/'$WORKERCORES'/g' \
    -e 's/$QUEUENAME/'$QUEUENAME'/g' \
    -e 's/$NAME/'G2_SD'/g' \
    -e 's/$SIMULATION/'ANNNP_G2_SD.py'/g' job_script.sh > temp.sh
qsub -V temp.sh
#########################################################################


########################## Extensive but light load #####################
# export NUMCORES=32 
# export WORKERCORES=1
# export WORKERS=32

# sed -e 's/$NUMCORES/'$NUMCORES'/g' \
#     -e 's/$WORKERCORES/'$WORKERCORES'/g' \
#     -e 's/$QUEUENAME/'$QUEUENAME'/g' \
#     -e 's/$NAME/'FSS_G2'/g' \
#     -e 's/$SIMULATION/'ANNNP_FSS_G2.py'/g' job_script.sh > temp.sh
# qsub -V temp.sh
#########################################################################


# # ######################### Expensive load ##############################
# export NUMCORES=32 
# export WORKERCORES=4
# export WORKERS=8

# sed -e 's/$NUMCORES/'$NUMCORES'/g' \
#     -e 's/$WORKERCORES/'$WORKERCORES'/g' \
#     -e 's/$QUEUENAME/'$QUEUENAME'/g' \
#     -e 's/$NAME/'FSS_G2'/g' \
#     -e 's/$SIMULATION/'ANNNP_FSS_C2.py'/g' job_script.sh > temp.sh
# qsub -V temp.sh
# #########################################################################

rm -r temp.sh