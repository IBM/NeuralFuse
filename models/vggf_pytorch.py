import torch
from torch import nn

from faultinjection_ops import zs_faultinjection_ops
from quantized_ops import zs_quantized_ops

# Per layer clamping currently based on manual values set
weight_clamp_values = [
    0.2,
    0.2,
    0.15,
    0.13,
    0.1,
    0.1,
    0.1,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
]
fc_weight_clamp = 0.1


class VGGPy(nn.Module):
    def __init__(self, features, classifier, classes=10, init_weights=True):
        super(VGGPy, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.classifier = classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, classes),
        # )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_classifierPy(
    classes,
    precision,
    ber,
    position,
    faulty_layers,
):
    if "linear" in faulty_layers:
        classifier = nn.Sequential(
            zs_faultinjection_ops.nnLinearPerturbWeight_op(
                25088,
                4096,
                precision,
                fc_weight_clamp,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            zs_faultinjection_ops.nnLinearPerturbWeight_op(
                4096,
                4096,
                precision,
                fc_weight_clamp,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            zs_faultinjection_ops.nnLinearPerturbWeight_op(
                4096,
                classes,
                precision,
                fc_weight_clamp,
            )
        )
    else:
        classifier = nn.Sequential(
            zs_quantized_ops.nnLinearSymQuant_op(
                25088, 4096, precision, fc_weight_clamp
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            zs_quantized_ops.nnLinearSymQuant_op(
                4096, 4096, precision, fc_weight_clamp
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            zs_quantized_ops.nnLinearSymQuant_op(
                4096, classes, precision, fc_weight_clamp
            )
        )

    return classifier


def make_layersPy(
    cfg,
    in_channels,
    batch_norm,
    precision,
    ber,
    position,
    faulty_layers,
):
    layers = []
    # in_channels = 3
    cl = 0
    # pdb.set_trace()
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if "conv" in faulty_layers:
                conv2d = zs_faultinjection_ops.nnConv2dPerturbWeight_op(
                    in_channels,
                    v,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    precision=precision,
                    clamp_val=weight_clamp_values[cl],
                )
            else:
                conv2d = zs_quantized_ops.nnConv2dSymQuant_op(
                    in_channels,
                    v,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    precision=precision,
                    clamp_val=weight_clamp_values[cl],
                )
            cl = cl + 1

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        "M",
        512,
        512,
        "M",
        512,
        512,
        "M",
    ],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def vggfPy(
    cfg,
    input_channels,
    classes,
    batch_norm,
    precision,
    ber,
    position,
    faulty_layers,
):
    model = VGGPy(
        make_layersPy(
            cfgs[cfg],
            in_channels=input_channels,
            batch_norm=batch_norm,
            precision=precision,
            ber=ber,
            position=position,
            faulty_layers=faulty_layers,
        ),
        make_classifierPy(
            classes,
            precision,
            ber,
            position,
            faulty_layers,
        ),
        classes,
        True,
    )
    return model