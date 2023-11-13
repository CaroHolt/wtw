#!/bin/sh
export CUDA_VISIBLE_DEVICES=3

for ADAPTER in '15_www.dailymail.co.uk' '78_www.wired.com' '17_www.express.co.uk' '45_www.npr.org' '81_www.librarything.com' \
'37_www.instructables.com' '91_www.entrepreneur.com' '11_link.springer.com' '70_www.insiderpages.com' '100_www.ign.com' '10_www.eventbrite.com' \
'22_forums.macrumors.com' '77_www.androidheadlines.com' '84_www.glassdoor.com' '85_www.pcworld.com' '60_www.csmonitor.com' \
'99_www.lonelyplanet.com' '39_www.booking.com' 'journals.plos.org' 'www.frontiersin.org' '58_medium.com'; do

  LEARNING_RATE=1e-4
  SEED=5
  MODEL_NAME=microsoft/deberta-base

  OUTPUT_DIR="../adapters/${ADAPTER}_${MODEL_NAME}_${LEARNING_RATE}_${SEED}/"

  python3 run_mlm_adapter.py \
      --model_name_or_path ${MODEL_NAME} \
      --train_file ./corpora/${ADAPTER}.train.json \
      --validation_file ./corpora/${ADAPTER}.val.json \
      --do_train \
      --do_eval \
      --output_dir ${OUTPUT_DIR} \
      --overwrite_output_dir \
      --num_train_epochs 20 \
      --adapter_config "bapnafi"\
      --gradient_accumulation_steps 5 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 50 \
      --seed ${SEED} \
      --learning_rate ${LEARNING_RATE} \
      --train_adapter \
      --cache_dir ./cache_dir/ \


done;
