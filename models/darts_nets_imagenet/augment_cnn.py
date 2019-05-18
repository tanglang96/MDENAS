""" CNN for network augmentation """
from models.darts_nets_imagenet.augment_cells import AugmentCell
from models.darts_nets_imagenet import ops
import torch.nn.functional as F
from modules.layers import *
from utils import *


class AuxiliaryHeadImageNet(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AugmentCNNImageNet(MyNetwork):
    def __init__(self, C_in=3, C=48, num_classes=1000, layers=14, auxiliary=False, genotype=None, drop_out=0.0,
                 bn_param=(0.1, 1e-3)):
        super(AugmentCNNImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.C_in = C_in
        self.C = C
        self.genotype = genotype
        self.drop_out = drop_out
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = AugmentCell(genotype, C_prev_prev, C_prev, C_curr, reduction_prev, reduction)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        if self.drop_out > 0:
            out = F.dropout(out.view(out.size(0), -1), training=self.training, p=self.drop_out)
        logits = self.classifier(out)
        if self._auxiliary and self.training:
            return logits, logits_aux
        else:
            return logits

    @property
    def config(self):
        return {
            'name': AugmentCNNImageNet.__name__,
            'bn': self.get_bn_param(),
            'drop_out': self.drop_out,
            'C_in': self.C_in,
            'C': self.C,
            'layers': self._layers,
            'auxiliary': self._auxiliary,
            'gene': str(self.genotype)
        }
