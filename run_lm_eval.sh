#!/bin/bash

DIR_PATH="/e/scratch/reformo/porian1/checkpoints/modded-nanogpt-moe/26-04-29-large_scale_baseline/000_26-04-29-large_scale_baseline+/"

python hellaswag.py \
    --run_dir ${DIR_PATH} \
    --tasks blimp,lambada_openai,arc_easy,arc_challenge,piqa,hellaswag,winogrande,glue \
    --limit 10 > ${DIR_PATH}/logfile_eval.log 2>&1