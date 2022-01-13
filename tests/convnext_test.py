import pandas as pd 
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
import convnextpl 

cifar = CIFAR10DataModule()
model = convnextpl.ConvNeXt()

trainer = Trainer()
trainer.fit(model, cifar)