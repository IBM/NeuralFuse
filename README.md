### Based model training:
```python
python zs_main.py [resnet18 | resnet50 | vgg11 | vgg16 | vgg19] train [cifar10 | gtsrb | cifar100] -E 300 -LR 0.001 -BS 256 
```

### NerualFuse training:
```python
python zs_main.py [resnet18 | resnet50 | vgg11 | vgg16 | vgg19] transform_eopm_gen [cifar10 | gtsrb | cifar100] -ber 0.01 -cp [please input the model path here] -E 300 -LR 0.001 -BS 256 -LM 5 -N 10 -G [ConvL | ConvS | DeConvL | DeConvS | UNetL | UNetS]
```

### Evaluation: Set setting in config.py first. (clean / generator_base)
```python
python zs_main.py [resnet18 | resnet50 | vgg11 | vgg16 | vgg19] transform_eval [cifar10 | gtsrb | cifar100] -ber 0.01 -cp [please input the model path here] -BS 256 -TBS 256 -G [ConvL | ConvS | DeConvL | DeConvS | UNetL | UNetS]
```

#### Choose Dataset:
* cifar10、cifar100、gtsrb、imagenet224