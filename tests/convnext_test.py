import comet_ml
import pandas as pd 
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
from convnextpl import Convnext
import pathlib, os 

here = pathlib.Path(__file__).parent.absolute()
key = [f.rstrip() for f in open(os.path.join(here, 'credentials'))][0]

cifar = CIFAR10DataModule()
num_classes = 10

model = Convnext(num_classes=num_classes)

cometlogger = CometLogger(
    api_key = key,
    project_name = 'convnext-test',
    workspace =  'jlehrer1',
)

trainer = Trainer(
    logger=cometlogger,
    max_epochs=100000,
)

trainer.fit(model, cifar)