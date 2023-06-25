# NeuralFuse

Official repo to reproduce the paper "Neuralfuse: Learning to Improve the Accuracy of Access-Limited Neural Network Inference in Low-Voltage Regimes"

## Usage

## Run Base Model Training with QAT:
```python
python zs_main.py [resnet18 | resnet50 | vgg11 | vgg16 | vgg19] train [cifar10 | gtsrb | cifar100 | imagenet224] -E 300 -LR 0.001 -BS 256 
```

## Run NerualFuse Training:
```python
python zs_main.py [resnet18 | resnet50 | vgg11 | vgg16 | vgg19] transform_eopm_gen [cifar10 | gtsrb | cifar100 | imagenet224] -ber 0.01 -cp [please input the model path here] -E 300 -LR 0.001 -BS 256 -LM 5 -N 10 -G [ConvL | ConvS | DeConvL | DeConvS | UNetL | UNetS]
```

## Run NerualFuse Evaluation with Perturbed Base Model: 
Please set the setting in config.py first. 

For example, in config.py:

```python
'''
cfg.testing_mode: Choosing the evaluation mode.
    1. clean: Evaluate the clean accuracy of specific base model. 
    2. generator_base: Evalueate the improved accuracy by NeuralFuse on specific perturbed base model.
cfg.G_PATH: The path of the NeuralFuse model.
'''
cfg.testing_mode = 'generator_base' # clean / generator_base
cfg.G_PATH = ''  # Only work in generator_base mode.
```

```python
python zs_main.py [resnet18 | resnet50 | vgg11 | vgg16 | vgg19] transform_eval [cifar10 | gtsrb | cifar100 | imagenet224] -ber 0.01 -cp [please input the model path here] -BS 256 -TBS 256 -G [ConvL | ConvS | DeConvL | DeConvS | UNetL | UNetS]
```

## Arguments:
TBA.

## Choose Dataset:
* cifar10、cifar100、gtsrb、imagenet10

## Notes:
We also adopt the Pytorch offical architecture settings for all of the base models. To use the version of Pytorch offical implementation, please change the args into [resnet18Py | resnet50Py | vgg11Py | vgg16Py | vgg19Py] instead. 