#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR='./outputdir'
EVAL_FILE='./evaluation_results.txt'

for TOPK in 1 2 3 4 5 6 7 8 9 10; do

  for SEED in 5 10 42 88; do

    for WEIGHTING in 'entropy' 'prior' 'tfidf' 'sent_sim' 'uniform'; do

      for COMBI in 'average' 'ensemble'; do

        for EVAL in '27_www.reuters.com.test.json' '93_techcrunch.com.test.json' '89_www.fastcompany.com.test.json' '94_www.nme.com.test.json' '46_www.fool.com.test.json' '71_www.inquisitr.com.test.json' \
          '50_mashable.com.test.json' '47_www.tripadvisor.com.test.json' '16_www.ncbi.nlm.nih.gov.test.json' 'yelp_sample_53M.json.test.json'; do


          python3 run_clm_adapter.py \
            --model_name_or_path gpt2 \
            --validation_file ./corpora/${EVAL} \
            --do_eval \
            --output_dir ${OUTPUT_DIR} \
            --overwrite_output_dir \
            --per_device_eval_batch_size 32 \
            --combination_strategy ${COMBI} \
            --seed ${SEED} \
            --top_k ${TOPK} \
            --adapter_weighting ${WEIGHTING} \
            --eval_file ${EVAL_FILE} \
            --adapter_dir ../../../adapters/15_www.dailymail.co.uk/clm \
            ../../../adapters/78_www.wired.com/clm \
            ../../../adapters/17_www.express.co.uk/clm \
            ../../../adapters/45_www.npr.org/clm \
            ../../../adapters/81_www.librarything.com/clm \
            ../../../adapters/37_www.instructables.com/clm \
            ../../../adapters/91_www.entrepreneur.com/clm \
            ../../../adapters/11_link.springer.com/clm \
            ../../../adapters/70_www.insiderpages.com/clm \
            ../../../adapters/100_www.ign.com/clm \
            ../../../adapters/10_www.eventbrite.com/clm \
            ../../../adapters/22_forums.macrumors.com/clm \
            ../../../adapters/77_www.androidheadlines.com/clm \
            ../../../adapters/84_www.glassdoor.com/clm \
            ../../../adapters/85_www.pcworld.com/clm \
            ../../../adapters/60_www.csmonitor.com/clm \
            ../../../adapters/99_www.lonelyplanet.com/clm \
            ../../../adapters/39_www.booking.com/clm \
            ../../../adapters/journals.plos.org/clm \
            ../../../adapters/www.frontiersin.org/clm \
            ../../../adapters/58_medium.com/clm \
            --adapter_val_files ./corpora/15_www.dailymail.co.uk.val.json \
            ./corpora/78_www.wired.com.val.json \
            ./corpora/17_www.express.co.uk.val.json \
            ./corpora/45_www.npr.org.val.json \
            ./corpora/81_www.librarything.com.val.json \
            ./corpora/37_www.instructables.com.val.json \
            ./corpora/91_www.entrepreneur.com.val.json \
            ./corpora/11_link.springer.com.val.json \
            ./corpora/70_www.insiderpages.com.val.json \
            ./corpora/100_www.ign.com.val.json \
            ./corpora/10_www.eventbrite.com.val.json \
            ./corpora/22_forums.macrumors.com.val.json \
            ./corpora/77_www.androidheadlines.com.val.json \
            ./corpora/84_www.glassdoor.com.val.json \
            ./corpora/85_www.pcworld.com.val.json \
            ./corpora/60_www.csmonitor.com.val.json \
            ./corpora/99_www.lonelyplanet.com.val.json \
            ./corpora/39_www.booking.com.val.json \
            ./corpora/journals.plos.org.val.json \
            ./corpora/www.frontiersin.org.val.json \
            ./corpora/58_medium.com.val.json

        done

      done

    done

  done

done
