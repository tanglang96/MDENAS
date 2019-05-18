#!/usr/bin/env bash
chmod +x /userhome/config.sh
bash /userhome/config.sh
cd /userhome/code/MDENAS
python3 run_exp.py --path Exp/mobile_cpu --dataset 'imagenet' --model_type "cpu" --mobile_gene "[(5, 3), (7, 1), (3, 6), (5, 3), (5, 3), (5, 3), (5, 1), (5, 6), (5, 1), (5, 6), (7, 3), (5, 3), (7, 3), (7, 6), (3, 6), (5, 3), (7, 6), (7, 1), (3, 3), (3, 3)]"
