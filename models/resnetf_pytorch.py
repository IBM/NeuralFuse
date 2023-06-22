"""ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import torch.nn.functional as F
from torch import nn

from faultinjection_ops import zs_faultinjection_ops
from quantized_ops import zs_quantized_ops

conv_clamp_val = 0.05
fc_clamp_val = 0.1


class BasicBlockPy(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride,
        precision,
        ber,
        position,
        faulty_layers,
    ):
        super(BasicBlockPy, self).__init__()
        if "conv" in faulty_layers:
            # print('In block')
            self.conv1 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        if "conv" in faulty_layers:
            self.conv2 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv2 = zs_quantized_ops.nnConv2dSymQuant_op(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if "conv" in faulty_layers:
                self.downsample = nn.Sequential(
                    zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                        precision=precision,
                        clamp_val=conv_clamp_val,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.downsample = nn.Sequential(
                    zs_quantized_ops.nnConv2dSymQuant_op(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                        precision=precision,
                        clamp_val=conv_clamp_val,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        return out

class BottleneckPy(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride,
        precision,
        ber,
        position,
        faulty_layers,
    ):
        super(BottleneckPy, self).__init__()
        if "conv" in faulty_layers:
            # print('In block')
            self.conv1 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                in_planes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_planes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.bn1 = nn.BatchNorm2d(planes)
        if "conv" in faulty_layers:
            self.conv2 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv2 = zs_quantized_ops.nnConv2dSymQuant_op(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.bn2 = nn.BatchNorm2d(planes)
        if "conv" in faulty_layers:
            self.conv3 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                planes,
                self.expansion * planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv3 = zs_quantized_ops.nnConv2dSymQuant_op(
                planes,
                self.expansion * planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if "conv" in faulty_layers:
                self.downsample = nn.Sequential(
                    zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                        precision=precision,
                        clamp_val=conv_clamp_val,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.downsample = nn.Sequential(
                    zs_quantized_ops.nnConv2dSymQuant_op(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        bias=False,
                        precision=precision,
                        clamp_val=conv_clamp_val,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu3(out)
        return out


class ResNetPy(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes,
        precision,
        ber,
        position,
        faulty_layers,
    ):
        super(ResNetPy, self).__init__()
        self.in_planes = 64

        if "conv" in faulty_layers:
            # print('In first')
            self.conv1 = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        else:
            self.conv1 = zs_quantized_ops.nnConv2dSymQuant_op(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                precision=precision,
                clamp_val=conv_clamp_val,
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = m = nn.MaxPool2d(
            kernel_size=3, 
            stride=2,
            padding=1,
            dilation=1)
        self.layer1 = self._make_layerPy(
            block,
            64,
            num_blocks[0],
            stride=1,
            precision=precision,
            ber=ber,
            position=position,
            faulty_layers=faulty_layers,
        )
        self.layer2 = self._make_layerPy(
            block,
            128,
            num_blocks[1],
            stride=2,
            precision=precision,
            ber=ber,
            position=position,
            faulty_layers=faulty_layers,
        )
        self.layer3 = self._make_layerPy(
            block,
            256,
            num_blocks[2],
            stride=2,
            precision=precision,
            ber=ber,
            position=position,
            faulty_layers=faulty_layers,
        )
        self.layer4 = self._make_layerPy(
            block,
            512,
            num_blocks[3],
            stride=2,
            precision=precision,
            ber=ber,
            position=position,
            faulty_layers=faulty_layers,
        )
        if "linear" in faulty_layers:
            self.fc = zs_faultinjection_ops.nnLinearPerturbWeight_op(
                512 * block.expansion,
                num_classes,
                precision,
                fc_clamp_val,
            )
        else:
            self.fc = zs_quantized_ops.nnLinearSymQuant_op(
                512 * block.expansion, num_classes, precision, fc_clamp_val
            )

    def _make_layerPy(
        self,
        block,
        planes,
        num_blocks,
        stride,
        precision,
        ber,
        position,
        faulty_layers,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    precision,
                    ber,
                    position,
                    faulty_layers,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18Py(
    classes,
    precision,
    ber,
    position,
    faulty_layers,
):
    return ResNetPy(
        BasicBlockPy,
        [2, 2, 2, 2],
        classes,
        precision,
        ber,
        position,
        faulty_layers,
    )


def ResNet34Py(
    classes,
    precision,
    ber,
    position,
    faulty_layers,
):
    return ResNetPy(
        BasicBlockPy,
        [3, 4, 6, 3],
        classes,
        precision,
        ber,
        position,
        faulty_layers,
    )

def ResNet50Py(
    classes,
    precision,
    ber,
    position,
    faulty_layers,
):
    return ResNetPy(
        BottleneckPy,
        [3, 4, 6, 3],
        classes,
        precision,
        ber,
        position,
        faulty_layers,
    )

def ResNet101Py(
    classes,
    precision,
    ber,
    position,
    faulty_layers,
):
    return ResNetPy(
        BottleneckPy,
        [3, 4, 23, 3],
        classes,
        precision,
        ber,
        position,
        faulty_layers,
    )


def resnetfPy(
    arch,
    classes,
    precision,
    ber,
    position,
    faulty_layers,
):
    if arch == "resnet18":
        return ResNet18Py(
            classes,
            precision,
            ber,
            position,
            faulty_layers,
        )
    elif arch == "resnet34":
        return ResNet34Py(
            classes,
            precision,
            ber,
            position,
            faulty_layers,
        )
    elif arch == "resnet50":
        return ResNet50Py(
            classes,
            precision,
            ber,
            position,
            faulty_layers,
        )
    elif arch == "resnet101":
        return ResNet101Py(
            classes,
            precision,
            ber,
            position,
            faulty_layers,
        )