#!/usr/bin/env bash
chmod +x /userhome/config.sh
bash /userhome/config.sh
cd /userhome/code/MDENAS
python3 run_exp.py --path Exp/darts_cifar --dataset 'cifar10' --dropout 0.2 --n_worker 4 --darts_gene "Genotype(normal=[[('sep_conv_5x5', 1), ('sep_conv_3x3', 0)], [('skip_connect', 0), ('sep_conv_5x5', 1)], [('sep_conv_5x5', 3), ('sep_conv_3x3', 1)], [('dil_conv_5x5', 3), ('max_pool_3x3', 4)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('sep_conv_5x5', 1)], [('skip_connect', 0), ('skip_connect', 1)], [('sep_conv_3x3', 3), ('skip_connect', 2)], [('dil_conv_3x3', 3), ('sep_conv_5x5', 0)]], reduce_concat=range(2, 6))"
