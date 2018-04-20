#!/bin/bash

# First argument: network-key
# Second argument: gpu
# Third argument: number of iterations 

export NAME=$(basename "$PWD")
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
BACKEND_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder-experiments/)
WORKDIR=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder-experiments/new_experiments)

nvidia-docker rm -f $NAME
rm snapshots/*

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -v /nrs/saalfeld:/nrs/saalfeld \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=${GUNPOWDER_PATH}:${BACKEND_PATH}:\$PYTHONPATH; python -u ${WORKDIR}/cremi_lr_affs.py $1 $3"
