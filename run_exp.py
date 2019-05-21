import argparse
import numpy as np
import os
import json
import torch

from models import *
from run_manager import RunManager
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='Exp/test')
parser.add_argument('--gpu', help='gpu available', default='0')
parser.add_argument('--manual_seed', default=0, type=int)

""" dataset """
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--test_batch_size', type=int, default=250)

""" train config """
parser.add_argument('--n_worker', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.3)

""" gene setting """
parser.add_argument("--model_type", default='gpu', type=str)
parser.add_argument("--darts_gene", default=None, type=str)
parser.add_argument("--mobile_gene", default=None, type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    os.makedirs(args.path, exist_ok=True)

    run_config = ImagenetRunConfig(
        **args.__dict__
    )

    if args.darts_gene is not None:
        if args.dataset == 'imagenet':
            from models.darts_nets_imagenet.augment_cnn import AugmentCNNImageNet

            net = AugmentCNNImageNet(num_classes=run_config.data_provider.n_classes, genotype=eval(args.darts_gene),
                                     drop_out=args.dropout)
        elif args.dataset == 'cifar10':
            from models.darts_nets_cifar.augment_cnn import AugmentCNN

            net = AugmentCNN(n_classes=run_config.data_provider.n_classes, genotype=eval(args.darts_gene),
                             drop_out=args.dropout)
    else:
        from models.normal_nets.proxyless_nets import proxyless_network

        gene = args.mobile_gene
        net = proxyless_network(
            structure=args.model_type, genotypes=eval(gene), n_classes=run_config.data_provider.n_classes,
            dropout_rate=args.dropout,
        )

    # build run manager
    run_manager = RunManager(args.path, net, run_config)

    run_manager.load_model()
    output_dict = {}

    print('Test on test set')
    loss, acc1, acc5 = run_manager.validate(is_test=True, return_top5=True)
    log = 'test_loss: %f\t test_acc1: %f\t test_acc5: %f' % (loss, acc1, acc5)
    run_manager.write_log(log, prefix='test')
    output_dict = {
        **output_dict,
        'test_loss': '%f' % loss, 'test_acc1': '%f' % acc1, 'test_acc5': '%f' % acc5,
        'total_params': '%f' % run_manager.total_params, 'gpu_latency': '%f' % run_manager.gpu_latency,
    }
    json.dump(output_dict, open('%s/test output' % args.path, 'w'), indent=4)
