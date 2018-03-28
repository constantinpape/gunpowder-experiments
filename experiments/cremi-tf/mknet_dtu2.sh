#!/bin/bash

export NAME="MK-UNET-DTU2"
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
CNNECTOME_PATH=$(readlink -f $HOME/Work/my_projects/nnets/CNNectome)
TRAIN_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf/)

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c "PYTHONPATH=${GUNPOWDER_PATH}:${CNNECTOME_PATH}:\$PYTHONPATH; python -u ${TRAIN_PATH}/mknet_dtu2.py"
