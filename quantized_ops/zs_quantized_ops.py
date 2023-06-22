"""
quantized version of nn.Linear
Followed the explanation in
https://pytorch.org/docs/stable/notes/extending.html on how to write a custom
autograd function and extend the nn.Linear class
https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html
"""

import torch
import torch.nn.functional as F
from config import cfg
from torch import nn
from torch.nn.modules.utils import _pair

# from torch.nn.modules.utils import _single

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


class SymmetricQuantizeDequantize(torch.autograd.Function):

    # Quantize and dequantize in the forward pass
    @staticmethod
    def forward(ctx, input, precision, clamp_val, use_max=True, q_method='symmetric_signed'):
        """
        Quantize and dequantize the model weights.
        The gradients will be applied to origin weights.
        :param ctx: Bulid-in parameter.
        :param input: Model weights.
        :param precision: The number of bits would be used to quantize the
                          model.
        :param clamp_val: The range to be used to clip the weights.
        """
        ctx.save_for_backward(input)
        # ctx.mark_dirty(input)

        if q_method == 'symmetric_unsigned' or q_method == 'asymmetric_unsigned':
            """
                Compute quantization step size.
                Mapping (min_val, max_val) linearly to (0, 2^m - 1)
            """
            if use_max and q_method == 'symmetric_unsigned':
                max_val = torch.max(torch.abs(input))
                min_val = -max_val
                input = torch.clamp(input, min_val, max_val)
                delta = max_val / (2 ** (precision - 1) - 1)
                input_q = torch.round((input / delta))
                input_q = input_q.to(torch.int32) + (2 ** (precision - 1) - 1)
                """
                    Dequantize introducing a quantization error in the data
                """
                input_dq = (input_q - (2 ** (precision - 1) - 1)) * delta
                input_dq = input_dq.to(torch.float32)

            elif not use_max and q_method == 'asymmetric_unsigned':
                max_val = torch.max(input)
                min_val = torch.min(input)
                delta = 1 / (2 ** (precision - 1) - 1)
                input = (input - min_val) / (max_val - min_val)
                input = input * 2 - 1 # To -1 ~ 1
                input_q = torch.round((input / delta))
                input_q = input_q.to(torch.int32) + (2 ** (precision - 1) - 1)
                """
                    Dequantize introducing a quantization error in the data
                """
                input_dq = (input_q - (2 ** (precision - 1) - 1)) * delta
                input_dq = ((input_dq + 1) / 2) * (max_val - min_val) + min_val
                input_dq = input_dq.to(torch.float32)
                

        elif q_method == 'symmetric_signed':
            """
                Compute quantization step size.
                Mapping (-max_val, max_val) linearly to (-127,127)
            """
            if use_max:
                max_val = torch.max(torch.abs(input))
            else:
                max_val = clamp_val
    
            delta = max_val / (2 ** (precision - 1) - 1)
            input_clamped = torch.clamp(input, -max_val, max_val)
            input_q = torch.round((input_clamped / delta))
            if precision > 0 and precision <= 8:
                input_q = input_q.to(torch.int8)
            elif precision == 16:
                input_q = input_q.to(torch.int16)
            else:
                input_q = input_q.to(torch.int32)
    
            """
                Dequantize introducing a quantization error in the data
            """
            input_dq = input_q * delta
            input_dq = input_dq.to(torch.float32)
        # Return the dequantized weights tensor.
        # We want to update the original weights(not quantized weights) under
        # quantization aware training.
        # So, we don't use input.copy_(input_dq) to replace self.weight with
        # input_dq.

        return input_dq

    # Straight-through-estimator in backward pass
    # https://discuss.pytorch.org/t/
    #                       integrating-a-new-loss-function-to-autograd/3684/2
    # Without using this will cause gradients problems.
    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output, None, None


class nnLinearSymQuant(nn.Linear):
    """Applies a linear transformation to the incoming data: y = xA^T + b
    Along with the linear transform, the learnable weights are quantized and
    dequantized introducing a quantization error in the data.
    """

    def __init__(
        self, in_features, out_features, bias, precision=-1, clamp_val=0.1
    ):
        super().__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # if bias:
        #    self.bias = nn.Parameter(torch.Tensor(out_features))
        # else:
        #    self.register_parameter('bias', None)
        self.precision = precision
        self.clamp_val = clamp_val
        self.reset_parameters()

    def forward(self, input):
        if self.precision > 0:
            quantWeight = SymmetricQuantizeDequantize.apply
            weight = quantWeight(self.weight, self.precision, self.clamp_val)
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, precision={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.precision,
        )


def nnLinearSymQuant_op(in_features, out_features, precision, clamp_val):
    return nnLinearSymQuant(
        in_features, out_features, True, precision, clamp_val
    )


class nnConv2dSymQuant(nn.Conv2d):
    """
    Computes 2d conv output
    Weights are quantized and dequantized introducing a quantization error
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        precision=-1,
        clamp_val=0.5,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.precision = precision
        self.clamp_val = clamp_val

    def forward(self, input):
        if self.precision > 0:
            quantWeight = SymmetricQuantizeDequantize.apply
            quant_weight = quantWeight(
                self.weight, self.precision, self.clamp_val
            )
            
        return F.conv2d(
            input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def nnConv2dSymQuant_op(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    bias,
    precision,
    clamp_val,
):
    return nnConv2dSymQuant(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=1,
        bias=bias,
        padding_mode="zeros",
        precision=precision,
        clamp_val=clamp_val,
    )

class nnConvTranspose2dSymQuant(nn.ConvTranspose2d): 
    """
    Computes 2d convtranspose output
    Weights are quantized and dequantized introducing a quantization error
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        precision=-1,
        clamp_val=0.5,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.precision = precision
        self.clamp_val = clamp_val

    def forward(self, input):
        if self.precision > 0:
            quantWeight = SymmetricQuantizeDequantize.apply
            quant_weight = quantWeight(
                self.weight, self.precision, self.clamp_val
            )
            
        return F.conv_transpose2d(
            input = input,
            weight = quant_weight,
            bias = self.bias,
            stride = self.stride,
            padding = self.padding,
            groups = self.groups,
            dilation = self.dilation
        )

def nnConvTranspose2dSymQuant_op(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    bias,
    precision,
    clamp_val,
):
    return nnConvTranspose2dSymQuant(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=1,
        bias=bias,
        precision=precision,
        clamp_val=clamp_val,
    )
