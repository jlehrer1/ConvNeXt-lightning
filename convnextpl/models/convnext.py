import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from torch import Tensor 
from typing import List, Callable, Dict
from .blocks import Block, LayerNorm

class ConvNeXt(pl.LightningModule):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, 
        in_chans: int=3, 
        num_classes: int=1000, 
        depths: list=[3, 3, 9, 3], 
        dims: list=[96, 192, 384, 768], 
        drop_path_rate: int=0.,
        layer_scale_init_value: float=1e-6, 
        head_init_scale: float=1.,
        lr: float=1e-4,
        momentum: float=1e-4,
        weight_decay: float=1e-2,
        metrics: Dict[str, Callable] = {
            'acc' : accuracy
        },
        loss: Callable = F.cross_entropy,
        class_weights: Tensor = None
    ) -> None:
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

        self.lr = lr 
        self.momentum = momentum 
        self.weight_decay = weight_decay
        self.metrics = metrics
        self.loss = loss
        self.weights = class_weights
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
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
 