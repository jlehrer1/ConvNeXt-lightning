import pandas as pd 
import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule

cifar = CIFAR10DataModule()
