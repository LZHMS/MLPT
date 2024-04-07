#!/bin/bash

# custom config
TRAINER=$1
DATASET=$2
CFG=$3  # config file
CTP=$4  # class token position (end or middle)
NCTX=$5  # number of context tokens
SHOTS=$6  # number of shots (1, 2, 4, 8, 16)
CSC=$7  # class-specific context (False or True)
URBL=$8
FP=$9

python parse_test_res.py output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/GCE_${URBL}/${SHOTS}shots_${FP}noise/ \
       >> output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/parse_results.txt