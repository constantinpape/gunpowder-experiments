#!/bin/bash

# Inputs:
# 0 - model-path
# 1 - model-weights path
# 2 - output oath

# export NAME=$(basename "$PWD")
export NAME="resave-tf-model"
export USER_ID=${UID}

# Need to adapt this to own gunpowder 
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
WORKDIR=$(readlink -f .)

nvidia-docker run --rm \
    -u ${USER_ID} \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -v /nrs/saalfeld:/nrs/saalfeld \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre5 \
    /bin/bash -c "export CUDA_VISIBLE_DEVICES=0; PYTHONPATH=${GUNPOWDER_PATH}:\$PYTHONPATH; python -u ${WORKDIR}/resave_tf_model.py $1 $2 $3"
