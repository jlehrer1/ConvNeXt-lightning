import comet_ml
import pandas as pd 
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
import convnextpl 

cifar = CIFAR10DataModule()
num_classes = 10

model = convnextpl.ConvNeXt()

trainer = Trainer()
trainer.fit(model, cifar)