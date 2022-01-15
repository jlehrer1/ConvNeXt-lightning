# ConvNeXt-Lightning
An implementation of the ConvNeXt architecture built on the PyTorch-Lightning API. The base model code (forward passes and architecture) are from the FAIR (Facebook AI Research) repo [here](https://github.com/facebookresearch/ConvNeXt). The original paper, *A ConvNet for the 2020s* can be found [here](https://arxiv.org/abs/2201.03545)

This library allows easy loading of an untrained ConvNeXt model from the PyTorch Lightning API. Additionally, it provides an `ImageDataset(nn.Dataset)` module for image classification tasks. To install, run `pip install convnextpl`. 

We can let PyTorch-Lightning handle all of the training and input shapes with the `Trainer` module. Here is a minimum reproducable example of the `convnextpl` library.

```python
from pytorch_lightning import Trainer
from pl_bolts.datamodules import CIFAR10DataModule
from convnextpl import Convnext

cifar = CIFAR10DataModule()
num_classes = 10

model = Convnext( # leave all other parameters to their default
  lr=1e-4,
)

trainer = Trainer()
trainer.fit(model, cifar)
```

This will train a `ConvNeXt` CNN on the CIFAR 10 dataset!

This library is meant for quick implementation of the ConvNext architecture. The optimizer is SGD (not ADAM? Check [this](https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/) blogpost out!). Of course, if you want to use a different opimtizer or change which metrics are logged you can always subclass the `Convnext` class and override `configure_optimizers()`.

The Python repository can be found on the Python package index [here](https://pypi.org/project/convnextpl/0.0.1/)
