import torch 
import pytorch_lightning as pl
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from typing import List, Callable, Dict
from torch import Tensor 

from torchmetrics.functional import accuracy
import torch.nn.functional as F

from .blocks import LayerNorm, Block

class ConvNeXtIsotropic(pl.LightningModule):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
        https://arxiv.org/pdf/2201.03545.pdf
        Isotropic ConvNeXts (Section 3.3 in paper)
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depth (tuple(int)): Number of blocks. Default: 18.
        dims (int): Feature dimension. Default: 384
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, 
            in_chans=3, 
            num_classes=1000, 
            depth=18, 
            dim=384, 
            drop_path_rate=0., 
            layer_scale_init_value=0, 
            head_init_scale=1.,
            lr: float=1e-4,
            momentum: float=1e-4,
            weight_decay: float=1e-2,
            metrics: Dict[str, Callable]={
                'acc' : accuracy
            },
            loss: Callable = F.cross_entropy,
            class_weights: Tensor = None
        ):
        super().__init__()

        self.stem = nn.Conv2d(in_chans, dim, kernel_size=16, stride=16)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[Block(dim=dim, drop_path=dp_rates[i], 
                                    layer_scale_init_value=layer_scale_init_value)
                                    for i in range(depth)])

        self.norm = LayerNorm(dim, eps=1e-6) # final norm layer
        self.head = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        self.lr = lr 
        self.momentum = momentum 
        self.weight_decay = weight_decay
        self.metrics = metrics

        # Loss with class_weights, if they are passed 
        self.loss = loss
        self.weights = class_weights

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch 
        x = self.forward(x)
        loss = self.loss(x, y, self.weights)

        for metric, func in self.metrics.items():
            self.log(metric, func(x, y), logger=True)

        return loss 

    def validation_step(self, batch, batch_idx):
        x, y = batch 
        x = self.forward(x) 
        loss = self.loss(x, y, self.weights)

        for metric, func in self.metrics.items():
            self.log(metric, func(x, y), logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr, 
            momentum=self.momentum, 
            weight_decay=self.weight_decay,
        )

        return optimizer
