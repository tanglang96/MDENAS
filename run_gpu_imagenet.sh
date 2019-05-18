#!/usr/bin/env bash
chmod +x /userhome/config.sh
bash /userhome/config.sh
cd /userhome/code/MDENAS
python3 run_exp.py --path Exp/mobile_gpu --dataset 'imagenet' --model_type "gpu" --mobile_gene "[(7, 6), (5, 6), (7, 3), (3, 6), (5, 6), (3, 1), (7, 3), (7, 6), (3, 3), (3, 6), (7, 3), (3, 3), (3, 3), (5, 6)]"
