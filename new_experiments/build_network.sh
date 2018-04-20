#!/bin/bash

export NAME=$(basename "$PWD")
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
BACKEND_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder-experiments/)
WORKDIR=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder-experiments/new_experiments)

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c "PYTHONPATH=${GUNPOWDER_PATH}:${BACKEND_PATH}:\$PYTHONPATH; python -u ${WORKDIR}/build_network.py $1"
