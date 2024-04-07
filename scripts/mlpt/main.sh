#!/bin/bash

# custom config
DATA=data
TRAINER=$1

DATASET=$2
CFG=$3  # config file
CTP=$4  # class token position (end or middle)
NCTX=$5  # number of context tokens
SHOTS=$6  # number of shots (1, 2, 4, 8, 16)
CSC=$7  # class-specific context (False or True)
URBL=$8
FP=$9
FACTOR=${10}

for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/GCE${URBL}_Factor${FACTOR}/${SHOTS}shots_${FP}noise/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --load-epoch 50 \
        --use-robustloss \
        --eval-only \
        --model-dir ${DIR} \
        --num-fp ${FP} \
        TRAINER.MLPT.N_CTX ${NCTX} \
        TRAINER.MLPT.CSC ${CSC} \
        TRAINER.MLPT.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
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
            TRAINER.MLPT.N_CTX ${NCTX} \
            TRAINER.MLPT.CSC ${CSC} \
            TRAINER.MLPT.CLASS_TOKEN_POSITION ${CTP} \
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
            TRAINER.MLPT.N_CTX ${NCTX} \
            TRAINER.MLPT.CSC ${CSC} \
            TRAINER.MLPT.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
        fi
    fi
done