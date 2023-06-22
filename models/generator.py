import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from quantized_ops import zs_quantized_ops
from config import cfg

# ++++++++++++++++++++ Generator V1 ++++++++++++++++++++

class GeneratorConvLQ(nn.Module):
    """
    Apply reprogramming.
    """
    def __init__(self, precision):
        super(GeneratorConvLQ, self).__init__()

        if precision > 0:

            self.conv1_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 3, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn1_1 = nn.BatchNorm2d(32)
            self.relu1_1 = nn.ReLU()
            self.conv1_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn1_2 = nn.BatchNorm2d(32)
            self.relu1_2 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)
            self.conv2_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn2_1 = nn.BatchNorm2d(64)
            self.relu2_1 = nn.ReLU()
            self.conv2_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn2_2 = nn.BatchNorm2d(64)
            self.relu2_2 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)
            self.conv3_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn3_1 = nn.BatchNorm2d(128)
            self.relu3_1 = nn.ReLU()
            self.conv3_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn3_2 = nn.BatchNorm2d(128)
            self.relu3_2 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)
            self.conv4_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn4_1 = nn.BatchNorm2d(128)
            self.relu4_1 = nn.ReLU()
            self.upsample4_1 = nn.Upsample(scale_factor=2)
            self.conv4_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn4_2 = nn.BatchNorm2d(128)
            self.relu4_2 = nn.ReLU()
            self.conv5_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 128, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn5_1 = nn.BatchNorm2d(64)
            self.relu5_1 = nn.ReLU()
            self.upsample5_1 = nn.Upsample(scale_factor=2)
            self.conv5_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn5_2 = nn.BatchNorm2d(64)
            self.relu5_2 = nn.ReLU()
            self.conv6_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 32, 
                kernel_size = 3,
                stride = 1, 
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.relu6_1 = nn.ReLU()
            self.bn6_1 = nn.BatchNorm2d(32)
            self.upsample6_1 = nn.Upsample(scale_factor=2)
            self.conv6_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn6_2 = nn.BatchNorm2d(32)
            self.relu6_2 = nn.ReLU()
            self.convout = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 3, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bnout = nn.BatchNorm2d(3)
            self.tanh = torch.nn.Tanh()
        
        else:
            self.conv1_1 = nn.Conv2d(
                in_channels = 3, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn1_1 = nn.BatchNorm2d(32)
            self.relu1_1 = nn.ReLU()
            self.conv1_2 = nn.Conv2d(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn1_2 = nn.BatchNorm2d(32)
            self.relu1_2 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)
            self.conv2_1 = nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn2_1 = nn.BatchNorm2d(64)
            self.relu2_1 = nn.ReLU()
            self.conv2_2 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn2_2 = nn.BatchNorm2d(64)
            self.relu2_2 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)
            self.conv3_1 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn3_1 = nn.BatchNorm2d(128)
            self.relu3_1 = nn.ReLU()
            self.conv3_2 = nn.Conv2d(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn3_2 = nn.BatchNorm2d(128)
            self.relu3_2 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)
            self.conv4_1 = nn.Conv2d(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn4_1 = nn.BatchNorm2d(128)
            self.relu4_1 = nn.ReLU()
            self.upsample4_1 = nn.Upsample(scale_factor=2)
            self.conv4_2 = nn.Conv2d(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn4_2 = nn.BatchNorm2d(128)
            self.relu4_2 = nn.ReLU()
            self.conv5_1 = nn.Conv2d(
                in_channels = 128, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn5_1 = nn.BatchNorm2d(64)
            self.relu5_1 = nn.ReLU()
            self.upsample5_1 = nn.Upsample(scale_factor=2)
            self.conv5_2 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn5_2 = nn.BatchNorm2d(64)
            self.relu5_2 = nn.ReLU()
            self.conv6_1 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 32, 
                kernel_size = 3,
                stride = 1, 
                padding = 1,
                )
            self.relu6_1 = nn.ReLU()
            self.bn6_1 = nn.BatchNorm2d(32)
            self.upsample6_1 = nn.Upsample(scale_factor=2)
            self.conv6_2 = nn.Conv2d(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn6_2 = nn.BatchNorm2d(32)
            self.relu6_2 = nn.ReLU()
            self.convout = nn.Conv2d(
                in_channels = 32, 
                out_channels = 3, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bnout = nn.BatchNorm2d(3)
            self.tanh = torch.nn.Tanh()
        
    def forward(self, image):
        img = image.data.clone()
        # Encoder
        x = self.relu1_1(self.bn1_1(self.conv1_1(img)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.maxpool1(x)
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.maxpool2(x)
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.maxpool3(x)

        # Decoder
        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.upsample4_1(x)
        x = self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.relu5_1(self.bn5_1(self.conv5_1(x)))
        x = self.upsample5_1(x)
        x = self.relu5_2(self.bn5_2(self.conv5_2(x)))
        x = self.relu6_1(self.bn6_1(self.conv6_1(x)))
        x = self.upsample6_1(x)
        x = self.relu6_2(self.bn6_2(self.conv6_2(x)))
        x = self.bnout(self.convout(x))
        out = self.tanh(x)

        x_adv = torch.clamp(img + out, min=-1, max=1)

        return x_adv

# ++++++++++++++++++++ Generator ConvSmall ++++++++++++++++++++

class GeneratorConvSQ(nn.Module):
    """
    Apply reprogramming.
    """
    def __init__(self, precision):
        super(GeneratorConvSQ, self).__init__()

        if precision > 0:
            self.conv1_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 3, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn1_1 = nn.BatchNorm2d(32)
            self.relu1_1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn2_1 = nn.BatchNorm2d(64)
            self.relu2_1 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

            self.conv3_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn3_1 = nn.BatchNorm2d(64)
            self.relu3_1 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

            self.conv4_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn4_1 = nn.BatchNorm2d(64)
            self.relu4_1 = nn.ReLU()
            self.upsample4_1 = nn.Upsample(scale_factor=2)

            self.conv5_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn5_1 = nn.BatchNorm2d(32)
            self.relu5_1 = nn.ReLU()
            self.upsample5_1 = nn.Upsample(scale_factor=2)

            self.conv6_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 3, 
                kernel_size = 3,
                stride = 1, 
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.relu6_1 = nn.ReLU()
            self.bn6_1 = nn.BatchNorm2d(3)
            self.upsample6_1 = nn.Upsample(scale_factor=2)

            self.convout = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 3, 
                out_channels = 3, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bnout = nn.BatchNorm2d(3)
            self.tanh = torch.nn.Tanh()

        else:
            self.conv1_1 = nn.Conv2d(
                in_channels = 3, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn1_1 = nn.BatchNorm2d(32)
            self.relu1_1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2_1 = nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn2_1 = nn.BatchNorm2d(64)
            self.relu2_1 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

            self.conv3_1 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn3_1 = nn.BatchNorm2d(64)
            self.relu3_1 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

            self.conv4_1 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn4_1 = nn.BatchNorm2d(64)
            self.relu4_1 = nn.ReLU()
            self.upsample4_1 = nn.Upsample(scale_factor=2)

            self.conv5_1 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn5_1 = nn.BatchNorm2d(32)
            self.relu5_1 = nn.ReLU()
            self.upsample5_1 = nn.Upsample(scale_factor=2)

            self.conv6_1 = nn.Conv2d(
                in_channels = 32, 
                out_channels = 3, 
                kernel_size = 3,
                stride = 1, 
                padding = 1,
                )
            self.relu6_1 = nn.ReLU()
            self.bn6_1 = nn.BatchNorm2d(3)
            self.upsample6_1 = nn.Upsample(scale_factor=2)

            self.convout = nn.Conv2d(
                in_channels = 3, 
                out_channels = 3, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bnout = nn.BatchNorm2d(3)
            self.tanh = torch.nn.Tanh()

    def forward(self, image):
        img = image.data.clone()
        # Encoder
        x = self.relu1_1(self.bn1_1(self.conv1_1(img)))
        x = self.maxpool1(x)
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.maxpool2(x)
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.maxpool3(x)

        # Decoder
        x = self.relu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.upsample4_1(x)
        x = self.relu5_1(self.bn5_1(self.conv5_1(x)))
        x = self.upsample5_1(x)
        x = self.relu6_1(self.bn6_1(self.conv6_1(x)))
        x = self.upsample6_1(x)
        x = self.bnout(self.convout(x))
        out = self.tanh(x)

        x_adv = torch.clamp(img + out, min=-1, max=1)

        return x_adv

# ++++++++++++++++++++ Generator DeConv Large ++++++++++++++++++++

class GeneratorDeConvLQ(nn.Module):
    """
    Apply reprogramming.
    """
    def __init__(self, precision):
        super(GeneratorDeConvLQ, self).__init__()

        if precision > 0:
            self.conv1_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 3, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn1_1 = nn.BatchNorm2d(32)
            self.relu1_1 = nn.ReLU()
            self.conv1_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn1_2 = nn.BatchNorm2d(32)
            self.relu1_2 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)
            self.conv2_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn2_1 = nn.BatchNorm2d(64)
            self.relu2_1 = nn.ReLU()
            self.conv2_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn2_2 = nn.BatchNorm2d(64)
            self.relu2_2 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)
            self.conv3_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn3_1 = nn.BatchNorm2d(128)
            self.relu3_1 = nn.ReLU()
            self.conv3_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn3_2 = nn.BatchNorm2d(128)
            self.relu3_2 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

            self.conv_mid = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn_mid = nn.BatchNorm2d(128)
            self.relu_mid = nn.ReLU()

            self.dconv4_1 = zs_quantized_ops.nnConvTranspose2dSymQuant_op(
                in_channels = 128, 
                out_channels = 64, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn4_1 = nn.BatchNorm2d(64)
            self.relu4_1 = nn.ReLU()
            self.conv4_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn4_2 = nn.BatchNorm2d(64)
            self.relu4_2 = nn.ReLU()
            self.dconv5_1 = zs_quantized_ops.nnConvTranspose2dSymQuant_op(
                in_channels = 64, 
                out_channels = 32, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn5_1 = nn.BatchNorm2d(32)
            self.relu5_1 = nn.ReLU()
            self.conv5_2 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn5_2 = nn.BatchNorm2d(32)
            self.relu5_2 = nn.ReLU()
            self.dconv6_1 = zs_quantized_ops.nnConvTranspose2dSymQuant_op(
                in_channels = 32, 
                out_channels = 3, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn6_1 = nn.BatchNorm2d(3)
            self.tanh = torch.nn.Tanh()        
        else:
            self.conv1_1 = nn.Conv2d(
                in_channels = 3, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn1_1 = nn.BatchNorm2d(32)
            self.relu1_1 = nn.ReLU()
            self.conv1_2 = nn.Conv2d(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn1_2 = nn.BatchNorm2d(32)
            self.relu1_2 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)
            self.conv2_1 = nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn2_1 = nn.BatchNorm2d(64)
            self.relu2_1 = nn.ReLU()
            self.conv2_2 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn2_2 = nn.BatchNorm2d(64)
            self.relu2_2 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)
            self.conv3_1 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn3_1 = nn.BatchNorm2d(128)
            self.relu3_1 = nn.ReLU()
            self.conv3_2 = nn.Conv2d(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn3_2 = nn.BatchNorm2d(128)
            self.relu3_2 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

            self.conv_mid = nn.Conv2d(
                in_channels = 128, 
                out_channels = 128, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn_mid = nn.BatchNorm2d(128)
            self.relu_mid = nn.ReLU()

            self.dconv4_1 = nn.ConvTranspose2d(
                in_channels = 128, 
                out_channels = 64, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                )
            self.bn4_1 = nn.BatchNorm2d(64)
            self.relu4_1 = nn.ReLU()
            self.conv4_2 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn4_2 = nn.BatchNorm2d(64)
            self.relu4_2 = nn.ReLU()
            self.dconv5_1 = nn.ConvTranspose2d(
                in_channels = 64, 
                out_channels = 32, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                )
            self.bn5_1 = nn.BatchNorm2d(32)
            self.relu5_1 = nn.ReLU()
            self.conv5_2 = nn.Conv2d(
                in_channels = 32, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn5_2 = nn.BatchNorm2d(32)
            self.relu5_2 = nn.ReLU()
            self.dconv6_1 = nn.ConvTranspose2d(
                in_channels = 32, 
                out_channels = 3, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                )
            self.bn6_1 = nn.BatchNorm2d(3)
            self.tanh = torch.nn.Tanh()        


    def forward(self, image):
        img = image.data.clone()
        # Encoder
        x = self.relu1_1(self.bn1_1(self.conv1_1(img)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.maxpool1(x)
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.maxpool2(x)
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.maxpool3(x)
        # Decoder
        x = self.relu_mid(self.bn_mid(self.conv_mid(x)))
        x = self.relu4_1(self.bn4_1(self.dconv4_1(x)))
        x = self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.relu5_1(self.bn5_1(self.dconv5_1(x)))
        x = self.relu5_2(self.bn5_2(self.conv5_2(x)))
        x = self.bn6_1(self.dconv6_1(x))
        out = self.tanh(x)

        x_adv = torch.clamp(image + out, min=-1, max=1)

        return x_adv


# ++++++++++++++++++++ Generator DeConv Small ++++++++++++++++++++

class GeneratorDeConvSQ(nn.Module):
    """
    Apply reprogramming.
    """
    def __init__(self, precision):
        super(GeneratorDeConvSQ, self).__init__()

        if precision > 0:
            self.conv1_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 3, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn1_1 = nn.BatchNorm2d(32)
            self.relu1_1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)
            self.conv2_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn2_1 = nn.BatchNorm2d(64)
            self.relu2_1 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)
            self.conv3_1 = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn3_1 = nn.BatchNorm2d(64)
            self.relu3_1 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)
            self.dconv4_1 = zs_quantized_ops.nnConvTranspose2dSymQuant_op(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn4_1 = nn.BatchNorm2d(64)
            self.relu4_1 = nn.ReLU()
            self.dconv5_1 = zs_quantized_ops.nnConvTranspose2dSymQuant_op(
                in_channels = 64, 
                out_channels = 32, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn5_1 = nn.BatchNorm2d(32)
            self.relu5_1 = nn.ReLU()
            self.dconv6_1 = zs_quantized_ops.nnConvTranspose2dSymQuant_op(
                in_channels = 32, 
                out_channels = 3, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                bias = True,
                precision = precision,
                clamp_val = None,
                )
            self.bn6_1 = nn.BatchNorm2d(3)
            self.tanh = torch.nn.Tanh()  
        else:
            self.conv1_1 = nn.Conv2d(
                in_channels = 3, 
                out_channels = 32, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn1_1 = nn.BatchNorm2d(32)
            self.relu1_1 = nn.ReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)
            self.conv2_1 = nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn2_1 = nn.BatchNorm2d(64)
            self.relu2_1 = nn.ReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)
            self.conv3_1 = nn.Conv2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 3, 
                stride = 1,
                padding = 1,
                )
            self.bn3_1 = nn.BatchNorm2d(64)
            self.relu3_1 = nn.ReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)
            self.dconv4_1 = nn.ConvTranspose2d(
                in_channels = 64, 
                out_channels = 64, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                )
            self.bn4_1 = nn.BatchNorm2d(64)
            self.relu4_1 = nn.ReLU()
            self.dconv5_1 = nn.ConvTranspose2d(
                in_channels = 64, 
                out_channels = 32, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                )
            self.bn5_1 = nn.BatchNorm2d(32)
            self.relu5_1 = nn.ReLU()
            self.dconv6_1 = nn.ConvTranspose2d(
                in_channels = 32, 
                out_channels = 3, 
                kernel_size = 4, 
                stride = 2,
                padding = 1,
                )
            self.bn6_1 = nn.BatchNorm2d(3)
            self.tanh = torch.nn.Tanh()       

    def forward(self, image):
        img = image.data.clone()
        # Encoder
        x = self.relu1_1(self.bn1_1(self.conv1_1(img)))
        x = self.maxpool1(x)
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.maxpool2(x)
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.maxpool3(x)
        # Decoder
        x = self.relu4_1(self.bn4_1(self.dconv4_1(x)))
        x = self.relu5_1(self.bn5_1(self.dconv5_1(x)))
        x = self.bn6_1(self.dconv6_1(x))
        out = self.tanh(x)

        x_adv = torch.clamp(image + out, min=-1, max=1)

        return x_adv
        
# ++++++++++++++++++++ GeneratorV14 UNet  ++++++++++++++++++++

class DoubleConvQ(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, precision, mid_channels=None):
        super().__init__()
        if precision > 0:
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                zs_quantized_ops.nnConv2dSymQuant_op(
                    in_channels = in_channels, 
                    out_channels = mid_channels, 
                    kernel_size = 3, 
                    stride = 1,
                    padding = 1,
                    bias = False,
                    precision = precision,
                    clamp_val = None,
                    ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                zs_quantized_ops.nnConv2dSymQuant_op(
                    in_channels = mid_channels, 
                    out_channels = out_channels, 
                    kernel_size = 3, 
                    stride = 1,
                    padding = 1,
                    bias = False,
                    precision = precision,
                    clamp_val = None,
                    ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
    def forward(self, x):
        return self.double_conv(x)

class DownQ(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, precision):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvQ(in_channels, out_channels, precision)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpQ(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, precision, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if precision > 0:
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConvQ(in_channels, out_channels, in_channels // 2, precision)
            else:
                self.up = zs_quantized_ops.nnConvTranspose2dSymQuant_op(
                    in_channels = in_channels, 
                    out_channels = in_channels // 2, 
                    kernel_size = 2, 
                    stride = 2,
                    padding = 1,
                    bias = True,
                    precision = precision,
                    clamp_val = None,
                    )
                self.conv = DoubleConvQ(in_channels, out_channels, precision)
        else:
            # if bilinear, use the normal convolutions to reduce the number of channels
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConvQ(in_channels, out_channels, in_channels // 2, precision)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConvQ(in_channels, out_channels, precision)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConvQ(nn.Module):
    def __init__(self, in_channels, out_channels, precision):
        super(OutConvQ, self).__init__()
        if precision > 0:
            self.conv = zs_quantized_ops.nnConv2dSymQuant_op(
                in_channels = in_channels, 
                out_channels = out_channels, 
                kernel_size = 1, 
                stride = 1,
                padding = 0,
                bias = True,
                precision = precision,
                clamp_val = None,
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class GeneratorUNetLQ(nn.Module):
    def __init__(self, precision, bilinear=False):
        super(GeneratorUNetLQ, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConvQ(3, 16, precision)
        self.down1 = DownQ(16, 32, precision)
        self.down2 = DownQ(32, 64, precision)
        factor = 2 if bilinear else 1
        self.down3 = DownQ(64, 128 // factor, precision)
        self.up1 = UpQ(128, 64 // factor, precision, bilinear)
        self.up2 = UpQ(64, 32 // factor, precision, bilinear)
        self.up3 = UpQ(32, 16 // factor, precision, bilinear)
        self.outc = OutConvQ(16, 3, precision)


    def forward(self, image):
        img = image.data.clone()
        x1 = self.inc(img) 
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x = self.up1(x4, x3) 
        x = self.up2(x, x2) 
        x = self.up3(x, x1) 
        out = self.outc(x)
        x_adv = torch.clamp(image + out, min=-1, max=1)
        return x_adv

class GeneratorUNetSQ(nn.Module):
    def __init__(self, precision, bilinear=False):
        super(GeneratorUNetSQ, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConvQ(3, 8, precision)
        self.down1 = DownQ(8, 16, precision)
        self.down2 = DownQ(16, 32, precision)
        factor = 2 if bilinear else 1
        self.down3 = DownQ(32, 64 // factor, precision)
        self.up1 = UpQ(64, 32 // factor, precision, bilinear)
        self.up2 = UpQ(32, 16 // factor, precision, bilinear)
        self.up3 = UpQ(16, 8 // factor, precision, bilinear)
        self.outc = OutConvQ(8, 3, precision)

    def forward(self, image):
        img = image.data.clone()
        x1 = self.inc(img) 
        x2 = self.down1(x1) 
        x3 = self.down2(x2) 
        x4 = self.down3(x3) 
        x = self.up1(x4, x3) 
        x = self.up2(x, x2) 
        x = self.up3(x, x1) 
        out = self.outc(x)
        x_adv = torch.clamp(image + out, min=-1, max=1)
        return x_adv