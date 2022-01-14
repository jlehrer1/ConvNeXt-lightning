# ConvNeXt-Lightning
An implementation of the ConvNeXt architecture built on the PyTorch-Lightning API. The base model code (forward passes and architecture) are from the FAIR (Facebook AI Research) repo [here](https://github.com/facebookresearch/ConvNeXt).

This library allows easy loading of an untrained ConvNeXt model from the PyTorch Lightning API. Additionally, it provides an `ImageDataset(nn.Dataset)` module for image classification tasks. 

We can let PyTorch-Lightning handle all of the training and input shapes with the `Trainer` module. Here is a minimum reproducable example of the `convnextpl` library.

```python
from pytorch_lightning import Trainer
from pl_bolts.datamodules import CIFAR10DataModule
import convnextpl 

cifar = CIFAR10DataModule()
num_classes = 10

model = convnextpl.ConvNeXt()

trainer = Trainer()
trainer.fit(model, cifar)
```

This will train a `ConvNeXt` CNN on the CIFAR 10 dataset!

The Python repository can be found on the Python package index [here](https://pypi.org/project/convnextpl/0.0.1/)