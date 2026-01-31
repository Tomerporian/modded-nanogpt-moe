#!/bin/bash

DIR_PATH="/data/mikey/exps/test/26-01-31-diff_no_softmax/000_26-01-31-diff_no_softmax+"

python hellaswag.py \
    --run_dir ${DIR_PATH} \
    --tasks wikitext,blimp,lambada_openai,arc_easy,piqa,hellaswag \
    --limit 10 > ${DIR_PATH}/logfile_eval.log 2>&1