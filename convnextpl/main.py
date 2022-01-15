from multiprocessing.sharedctypes import Value
from typing import List, Callable, Dict
from torchmetrics.functional import accuracy

from .data.imagedataset import ImageSet
from .models.convnext import ConvNeXt
from .models.convnext_isotropic import ConvNeXtIsotropic

def Convnext(
    type=None,
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
):
    if type == None:
        model = ConvNeXt(
            in_chans=in_chans,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            head_init_scale=head_init_scale,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            metrics=metrics
        )
    elif type == 'isotropic':
        model = ConvNeXtIsotropic(
            in_chans=in_chans,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            head_init_scale=head_init_scale,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            metrics=metrics
        )
    else:
        raise ValueError(f"Invalid value in type {type}. Must be one of [None, 'isotropic']")

    return model 