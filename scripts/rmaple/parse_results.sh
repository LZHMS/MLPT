#!/bin/bash

# custom config
TRAINER=$1

DATASET=$2
CFG=$3  # config file
NCTX=$4  # number of context tokens
PD=$5  # leanring depth for MaPLe
SHOTS=$6  # number of shots (1, 2, 4, 8, 16)
URBL=$7
FP=$8

python parse_test_res.py output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_depth${PD}/GCE_${URBL}/${SHOTS}shots_${FP}noise/ \
       >> output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/parse_results.txt