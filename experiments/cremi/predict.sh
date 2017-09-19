rm snapshots/*

export NAME=$(basename $PWD-prediction)
export USER_ID=${UID}
GUNPOWDER_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder)
PRED_PATH=$(readlink -f $HOME/Work/my_projects/nnets/gunpowder-experiments/experiments/cremi/)

nvidia-docker rm -f $NAME

nvidia-docker run --rm \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/saalfeld/home/papec:/groups/saalfeld/home/papec \
    -w /workspace \
    --name $NAME \
    funkey/gunpowder:v0.3-pre2 \
    /bin/bash -c "PYTHONPATH=${GUNPOWDER_PATH}:\$PYTHONPATH; python -u ${PRED_PATH}/predict_affinities.py $1 $2 $3"
