"""
quantized version of nn.Linear and nn.Conv with bit errors injected in weights
Currently bit error injection is supported only in 8-bit type
Followed the explanation in
https://pytorch.org/docs/stable/notes/extending.html on how to write a
custom autograd function and extend the nn.Linear class
https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html
"""

from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch import nn
import torch

import copy
import math

import faultsMap as fmap

debug = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32


class FaultInject(torch.autograd.Function):
    """
    Perturb the model weights.
    Later, quantize and dequantize the model weights in the forward pass.
    """

    @staticmethod
    def forward(ctx, input, precision, clamp_val, BitErrorMap0to1, BitErrorMap1to0, use_max=True, q_method='symmetric_signed'):
        ctx.save_for_backward(input)
        # ctx.mark_dirty(input)

        """
        Compute quantization step size. Mapping (-max_val, max_val)
        linearly to (-127,127)
        """

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
                input_q = input_q.to(torch.int32) + (2 ** (precision - 1) - 1) # [0, 255] => 0....011111111 for 32 bits
            elif not use_max and q_method == 'asymmetric_unsigned':
                max_val = torch.max(input)
                min_val = torch.min(input)
                delta = 1 / (2 ** (precision - 1) - 1)
                input = (input - min_val) / (max_val - min_val)
                input = input * 2 - 1 # To -1 ~ 1
                input_q = torch.round((input / delta))
                input_q = input_q.to(torch.int32) + (2 ** (precision - 1) - 1) # [0, 255] => 0....011111111 for 32 bits

        elif q_method == 'symmetric_signed':
            """
                Compute quantization step size.
                Mapping (-max_val, max_val) linearly to (-127,127)
            """
            # fix me : Need to add a parameter for this one !
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
            Inject faults in the quantized weight
            as determined by the bit error map
            
            BitErrorMap0, BitErrorMap1 = self.MapWeightsToBitErrors(numWeights)
            convert to int8, since this becomes uint8 by default
            after bitwise op with unsigned biterrormaps

        """

        if q_method == 'symmetric_unsigned' or q_method == 'asymmetric_unsigned':
            input_qand = torch.bitwise_and(BitErrorMap1to0, input_q)
            input_qor = torch.bitwise_or(BitErrorMap0to1, input_qand)
        elif q_method == 'symmetric_signed':
            input_qand = torch.bitwise_and(BitErrorMap1to0, input_q).to(torch.int8)
            input_qor = torch.bitwise_or(BitErrorMap0to1, input_qand).to(torch.int8)


        """
            Dequantize introducing a quantization error in the
            data along with the weight perturbation
        """
        if q_method == 'symmetric_unsigned':
            input_dq = (input_qor - (2 ** (precision - 1) - 1)) * delta
            input_dq = torch.clamp(input_dq, min_val, max_val)
            input_dq = input_dq.to(torch.float32)
        elif q_method == 'asymmetric_unsigned':
            input_dq = (input_qor - (2 ** (precision - 1) - 1)) * delta
            input_dq = ((input_dq + 1) / 2) * (max_val - min_val) + min_val
            input_dq = torch.clamp(input_dq, min_val, max_val)
            input_dq = input_dq.to(torch.float32)
        elif q_method == 'symmetric_signed':
            input_dq = input_qor * delta
            input_dq = input_dq.to(torch.float32)

        """ For symmetric quantization, it is possible for dequantized tensors to get out of range compared to the original model"""

        input_dq_clamped = torch.clamp(input_dq, torch.min(input), torch.max(input))

        # Return the perturbed-dequantized weights tensor.
        # We want to perturb the weights just for one time.
        # So, we don't use input.copy_(input_dq) to replace
        # self.weight with input_dq.
        return input_dq

    # Straight-through-estimator in backward pass
    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output, None, None, None, None


class nnLinearPerturbWeight(nn.Linear):
    """Applies a linear transformation to the incoming data: y = xA^T + b
    Along with the linear transform, the learnable weights are quantized,
    and then a bit error perturbation is introduced. Then the weights are
    dequantized.

    """

    def __init__(self, in_features, out_features, bias, precision=-1, clamp_val=0.1):
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
            BitErrorMap0to1, BitErrorMap1to0 = self.genFaultMap(
                fmap.BitErrorMap0,
                fmap.BitErrorMap1,
                self.precision,
                self.weight,
            )
            perturbweight = FaultInject.apply
            perturbed_weights = perturbweight(
                self.weight,
                self.precision,
                self.clamp_val,
                BitErrorMap0to1,
                BitErrorMap1to0,
            )
        if debug:
            print(torch.min(self.weight).detach().cpu().numpy(), torch.max(self.weight).detach().cpu().numpy())
            print(torch.min(perturbed_weights).detach().cpu().numpy(), torch.max(perturbed_weights).detach().cpu().numpy())
        return F.linear(input, perturbed_weights, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}, precision={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            self.precision,
        )

    def genFaultMap(self, BitErrorMap_flip0to1, BitErrorMap_flip1to0, precision, weights):

        numweights = torch.numel(weights)

        mem_array_rows = BitErrorMap_flip0to1.shape[0]
        mem_array_cols = BitErrorMap_flip0to1.shape[1]

        weights_per_row = math.floor(mem_array_cols / precision)

        # Because in random bit error, BitErrorMap0to1 and BitErrorMap1to0 will be the same for every layers in the same perturbed model
        # Therefore, calculate the for loop for once is enough.
        if fmap.BitErrorMap0to1 is None and fmap.BitErrorMap1to0 is None:
            BitErrorMap0to1 = torch.zeros(
                (mem_array_rows, weights_per_row), dtype=torch.uint8
            ).to(device)
            BitErrorMap1to0 = torch.zeros(
                (mem_array_rows, weights_per_row), dtype=torch.uint8
            ).to(device)

            # Reshaping bit error map to map weights
            for k in range(0, weights_per_row):
                for j in range(0, precision):
                    BitErrorMap0to1[:, k] += (
                        BitErrorMap_flip0to1[:, k * precision + j] << j
                    )
                    BitErrorMap1to0[:, k] += (
                        BitErrorMap_flip1to0[:, k * precision + j] << j
                    )
            fmap.BitErrorMap0to1 = copy.deepcopy(BitErrorMap0to1)
            fmap.BitErrorMap1to0 = copy.deepcopy(BitErrorMap1to0)
        else:
            BitErrorMap0to1 = copy.deepcopy(fmap.BitErrorMap0to1)
            BitErrorMap1to0 = copy.deepcopy(fmap.BitErrorMap1to0)

        rows = math.ceil(numweights / weights_per_row)
        cols = weights_per_row
        num_banks = math.ceil(rows / mem_array_rows)
        BitErrorMap0to1 = torch.tile(BitErrorMap0to1, (num_banks, cols))
        # invert this one, since it needs to be And-ed
        BitErrorMap1to0 = torch.tile(~BitErrorMap1to0, (num_banks, cols))

        # This mapping is highly dependent on data flow
        BitErrorMap0to1 = BitErrorMap0to1.view(-1)[0:numweights]
        BitErrorMap1to0 = BitErrorMap1to0.view(-1)[0:numweights]

        BitErrorMap0to1 = torch.reshape(BitErrorMap0to1, weights.size())
        BitErrorMap1to0 = torch.reshape(BitErrorMap1to0, weights.size())

        return BitErrorMap0to1, BitErrorMap1to0


def nnLinearPerturbWeight_op(in_features, out_features, precision, clamp_val):
    return nnLinearPerturbWeight(
        in_features,
        out_features,
        True,
        precision,
        clamp_val,
    )


class nnConv2dPerturbWeight(nn.Conv2d):
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
            BitErrorMap0to1, BitErrorMap1to0 = self.genFaultMap(
                fmap.BitErrorMap0,
                fmap.BitErrorMap1,
                self.precision,
                self.weight,
            )
            perturbweight = FaultInject.apply
            perturbed_weights = perturbweight(
                self.weight,
                self.precision,
                self.clamp_val,
                BitErrorMap0to1,
                BitErrorMap1to0,
            )
        if debug:
            print(torch.min(self.weight).detach().cpu().numpy(), torch.max(self.weight).detach().cpu().numpy())
            print(torch.min(perturbed_weights).detach().cpu().numpy(), torch.max(perturbed_weights).detach().cpu().numpy())
        return F.conv2d(
            input,
            perturbed_weights,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def genFaultMap(self, BitErrorMap_flip0to1, BitErrorMap_flip1to0, precision, weights):
        
        numweights = torch.numel(weights)
        mem_array_rows = BitErrorMap_flip0to1.shape[0]
        mem_array_cols = BitErrorMap_flip0to1.shape[1]

        weights_per_row = math.floor(mem_array_cols / precision)

        # Because in random bit error, BitErrorMap0to1 and BitErrorMap1to0 will be the same for every layers in the same perturbed model
        # Therefore, calculate the for loop for once is enough.
        if fmap.BitErrorMap0to1 is None and fmap.BitErrorMap1to0 is None:
            BitErrorMap0to1 = torch.zeros(
                (mem_array_rows, weights_per_row), dtype=torch.uint8
            ).to(device)
            BitErrorMap1to0 = torch.zeros(
                (mem_array_rows, weights_per_row), dtype=torch.uint8
            ).to(device)

            # Reshaping bit error map to map weights
            for k in range(0, weights_per_row):
                for j in range(0, precision):
                    BitErrorMap0to1[:, k] += (
                        BitErrorMap_flip0to1[:, k * precision + j] << j
                    )
                    BitErrorMap1to0[:, k] += (
                        BitErrorMap_flip1to0[:, k * precision + j] << j
                    )
            fmap.BitErrorMap0to1 = copy.deepcopy(BitErrorMap0to1)
            fmap.BitErrorMap1to0 = copy.deepcopy(BitErrorMap1to0)
        else:
            BitErrorMap0to1 = copy.deepcopy(fmap.BitErrorMap0to1)
            BitErrorMap1to0 = copy.deepcopy(fmap.BitErrorMap1to0)

        rows = math.ceil(numweights / weights_per_row)
        cols = weights_per_row
        num_banks = math.ceil(rows / mem_array_rows)
        BitErrorMap0to1 = torch.tile(BitErrorMap0to1, (num_banks, cols))
        # invert this one, since it needs to be And-ed
        BitErrorMap1to0 = torch.tile(~BitErrorMap1to0, (num_banks, cols))

        # This mapping is highly dependent on data flow
        BitErrorMap0to1 = BitErrorMap0to1.view(-1)[0:numweights]
        BitErrorMap1to0 = BitErrorMap1to0.view(-1)[0:numweights]

        BitErrorMap0to1 = torch.reshape(BitErrorMap0to1, weights.size())
        BitErrorMap1to0 = torch.reshape(BitErrorMap1to0, weights.size())

        return BitErrorMap0to1, BitErrorMap1to0


def nnConv2dPerturbWeight_op(in_channels, out_channels, kernel_size, stride, padding, bias, precision, clamp_val):
    return nnConv2dPerturbWeight(
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
