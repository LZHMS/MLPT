#!/bin/bash

# custom config
DATA=data
TRAINER=$1

DATASET=$2
CFG=$3  # config file
NCTX=$4  # number of context tokens
PD=$5  # leanring depth for MaPLe 
SHOTS=$6  # number of shots (1, 2, 4, 8, 16)
URBL=$7
FP=$8


for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_depth${PD}/GCE_${URBL}/${SHOTS}shots_${FP}noise/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        echo "Run this job and save the output to ${DIR}, with GCE ${URBL}"
        if [ "$URBL" = "True" ]; then
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --use-robustloss \
            --num-fp ${FP} \
            --prompt-depth ${PD}\
            TRAINER.MAPLE.N_CTX ${NCTX} \
            DATASET.NUM_SHOTS ${SHOTS}
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --num-fp ${FP} \
            --prompt-depth ${PD}\
            TRAINER.MAPLE.N_CTX ${NCTX} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
    fi
done