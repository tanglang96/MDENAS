import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time
from utils import *
import apex

class RunConfig:
    def __init__(self, dataset, test_batch_size, local_rank, world_size):

        self.dataset = dataset
        self.test_batch_size = test_batch_size
        self._data_provider = None
        self.local_rank = local_rank
        self.world_size = world_size
        self.print_frequency = 1

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    @property
    def data_config(self):
        raise NotImplementedError

    @property
    def data_provider(self):
        if self._data_provider is None:
            if self.dataset == 'imagenet':
                from data_providers.imagenet import ImagenetDataProvider
                self._data_provider = ImagenetDataProvider(**self.data_config)
            elif self.dataset == 'cifar10':
                from data_providers.cifar10 import CifarDataProvider
                self._data_provider = CifarDataProvider(**self.data_config)
            else:
                raise ValueError('do not support: %s' % self.dataset)
        return self._data_provider

    @data_provider.setter
    def data_provider(self, val):
        self._data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test


class RunManager:
    def __init__(self, path, net, run_config: RunConfig, out_log=True):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.out_log = out_log
        self.device = torch.device('cuda')
        self._logs_path, self._save_path = None, None
        self.best_acc = 0
        self.start_epoch = 0
        self.net = apex.parallel.convert_syncbn_model(nn.DataParallel(self.net)).cuda()
        self.print_net_info()
        self.criterion = nn.CrossEntropyLoss()
        cudnn.benchmark = True

    """ save path and log path """

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return self._logs_path

    def print_net_info(self):
        # parameters
        self.total_params = count_parameters(self.net)
        self.gpu_latency = self.get_gpu_latency()
        if self.out_log:
            print('Total training params: %.2fM' % (self.total_params / 1e6))
        net_info = {
            'param': '%.2fM' % (self.total_params / 1e6),
            'gpu latency': '%.2fms' % (self.gpu_latency),
        }

        with open('%s/net_info.txt' % self.logs_path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.net.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        if self.out_log:
            print("=> loading checkpoint '{}'".format(model_fname))

        if torch.cuda.is_available():
            checkpoint = torch.load(model_fname)
        else:
            checkpoint = torch.load(model_fname, map_location='cpu')

        self.net.load_state_dict(checkpoint['state_dict'])
        # set new manual seed
        new_manual_seed = int(time.time())
        torch.manual_seed(new_manual_seed)
        torch.cuda.manual_seed_all(new_manual_seed)
        np.random.seed(new_manual_seed)

        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        if 'best_acc' in checkpoint:
            self.best_acc = checkpoint['best_acc']

        if self.out_log:
            print("=> loaded checkpoint '{}'".format(model_fname))

    def save_config(self, print_info=True):
        """ dump run_config and net_config to the model_folder """
        os.makedirs(self.path, exist_ok=True)
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.net.module.config, open(net_save_path, 'w'), indent=4)
        if print_info:
            print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        if print_info:
            print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def write_log(self, log_str, prefix, should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    def validate(self, is_test=True, net=None, use_train_mode=False, return_top5=False):
        if self.run_config.dataset == 'imagenet':
            n_dataloader = 50000 // (self.run_config.test_batch_size * self.run_config.world_size) + 1
        elif self.run_config.dataset == 'cifar10':
            n_dataloader = 10000 // (self.run_config.test_batch_size * self.run_config.world_size) + 1
        if is_test:
            data_loader = self.run_config.test_loader
        else:
            data_loader = self.run_config.valid_loader

        if net is None:
            net = self.net

        if use_train_mode:
            net.train()
        else:
            net.eval()
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                if self.run_config.dataset == 'imagenet':
                    images, labels = data[0]["data"].cuda(async=True), data[0]["label"].squeeze().long().cuda(
                        async=True)
                elif self.run_config.dataset == 'cifar10':
                    images, labels = data[0].cuda(async=True), data[1].cuda(async=True)
                # compute output
                output = net(images)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_config.print_frequency == 0 or i + 1 == n_dataloader:
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    test_log = prefix + ': [{0}/{1}]\t' \
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                        format(i, n_dataloader - 1, batch_time=batch_time, loss=losses, top1=top1)
                    if return_top5:
                        test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    print(test_log)
        if return_top5:
            return losses.avg, top1.avg, top5.avg
        else:
            return losses.avg, top1.avg

    def get_gpu_latency(self):
        self.net.eval()
        latency = AverageMeter()
        for i in range(100):
            if self.run_config.dataset == 'imagenet':
                x = torch.randn(8, 3, 224, 224, device=self.device)
            elif self.run_config.dataset == 'cifar10':
                x = torch.randn(8, 3, 32, 32, device=self.device)
            with torch.no_grad():
                start = time.time()
                y = self.net(x)
            end = time.time()
            if i > 49:
                latency.update((end - start) * 1000, 1)
        return latency.avg
