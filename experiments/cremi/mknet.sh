#!/bin/bash

export NAME=$(basename "$PWD")
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
TRAIN_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/)

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre2 \
    /bin/bash -c "PYTHONPATH=${GUNPOWDER_PATH}:\$PYTHONPATH; python -u ${TRAIN_PATH}/mknet.py"
