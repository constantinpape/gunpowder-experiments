#!/bin/bash

# First argument: number of iterations, 
# Seconda argument: gpu

export NAME=$(basename "$PWD")
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
TRAIN_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi-tf/)

nvidia-docker rm -f $NAME
rm snapshots/*

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=${GUNPOWDER_PATH}:\$PYTHONPATH; python -u ${TRAIN_PATH}/training.py $1"
