import os
import torch

from .resnetf_pytorch import resnetfPy
from .resnetf import resnetf
from .resnet import resnet
from .vggf_pytorch import vggfPy
from .vggf import vggf
from .vgg import vgg

from faultmodels import randomfault
import faultsMap as fmap
from config import cfg

# Create the fault map from randomfault module.
def create_faults(precision, bit_error_rate, position, seed=0):
    rf = randomfault.RandomFaultModel(
        bit_error_rate, precision, position, seed=seed
    )
    fmap.BitErrorMap0 = (
        torch.tensor(rf.BitErrorMap_flip0).to(torch.int32).to(cfg.device)
    )
    fmap.BitErrorMap1 = (
        torch.tensor(rf.BitErrorMap_flip1).to(torch.int32).to(cfg.device)
    )


def init_models(arch, in_channels, precision, retrain, checkpoint_path, dataset='cifar10'):

    """
    Default model loader
    """
    classes = 10
    if dataset == 'cifar10':
        classes = 10
    elif dataset == 'cifar100':
        classes = 100
    elif dataset == 'imagenet':
        classes = 10
    elif dataset == 'gtsrb':
        classes = 43
    else:
        classes = 10

    if arch == "vgg11":
        if precision > 0:
            model = vggf("A", in_channels, classes, True, precision, 0, 0, [])
        else:
            model = vgg("A", in_channels, classes, True, precision)
    elif arch == "vgg16":
        if precision > 0:
            model = vggf("D", in_channels, classes, True, precision, 0, 0, [])
        else:
            model = vgg("D", in_channels, classes, True, precision)
    elif arch == "vgg19":
        if precision > 0:
            model = vggf("E", in_channels, classes, True, precision, 0, 0, [])
        else:
            model = vgg("E", in_channels, classes, True, precision)
    elif arch == "resnet18":
        if precision > 0:
            model = resnetf("resnet18", classes, precision, 0, 0, [])
        else:
            model = resnet("resnet18", classes, precision)
    elif arch == "resnet34":
        if precision > 0:
            model = resnetf("resnet34", classes, precision, 0, 0, [])
        else:
            model = resnet("resnet34", classes, precision)
    elif arch == "resnet50":
        if precision > 0:
            model = resnetf("resnet50", classes, precision, 0, 0, [])
        else:
            model = resnet("resnet50", classes, precision)
    elif arch == "resnet101":
        if precision > 0:
            model = resnetf("resnet101", classes, precision, 0, 0, [])
        else:
            model = resnet("resnet101", classes, precision)
    elif "resnet18Py" in arch:
        if precision > 0:
            model = resnetfPy("resnet18", classes, precision, ber=0, position=0, faulty_layers=[])
    elif "resnet50Py" in arch:
        if precision > 0:
            model = resnetfPy("resnet50", classes, precision, ber=0, position=0, faulty_layers=[])
    elif "vgg11Py" in arch:
        if precision > 0:
            model = vggfPy("A", 3, classes, True, precision, 0, 0, [])
    elif "vgg16Py" in arch:
        if precision > 0:
            model = vggfPy("D", 3, classes, True, precision, 0, 0, [])
    elif "vgg19Py" in arch:
        if precision > 0:
            model = vggfPy("E", 3, classes, True, precision, 0, 0, [])
    else:
        raise NotImplementedError

    # print(model)
    checkpoint_epoch = -1

    if retrain:
        if not os.path.exists(checkpoint_path):
            for x in range(cfg.epochs, -1, -1):
                if os.path.exists(model_path_from_base(checkpoint_path, x)):
                    checkpoint_path = model_path_from_base(checkpoint_path, x)
                    break

        if not os.path.exists(checkpoint_path):
            print("Checkpoint path not exists")
            return model, checkpoint_epoch

        # print("Restoring model from checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        # print("restored checkpoint at epoch - ", checkpoint["epoch"])
        # print("Training loss =", checkpoint["loss"])
        # print("Training accuracy =", checkpoint["accuracy"])
        checkpoint_epoch = checkpoint["epoch"]

    return model, checkpoint_epoch


def init_models_faulty(arch, in_channels, precision, retrain, checkpoint_path, faulty_layers, bit_error_rate, position, seed=0, dataset='cifar'):

    """
    Perturbed (if needed) model loader.
    """

    if not cfg.faulty_layers or len(cfg.faulty_layers) == 0:
        return init_models(
            arch, in_channels, precision, retrain, checkpoint_path
        )
    else:
        """Perturbed models, where the weights are injected with bit
        errors at the rate of ber at the specified positions"""
        
        classes = 10
        if dataset == 'cifar10':
            classes = 10
        elif dataset == 'cifar100':
            classes = 100
        elif dataset == 'gtsrb':
            classes = 43
        else:
            classes = 10 # Include ImageNet-10

        if arch == "vgg11":
            model = vggf(
                "A",
                in_channels,
                classes,
                True,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "vgg16":
            model = vggf(
                "D",
                in_channels,
                classes,
                True,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "vgg19":
            model = vggf(
                "E",
                in_channels,
                classes,
                True,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )    
        elif arch == "resnet18":
            model = resnetf(
                "resnet18",
                classes,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "resnet34":
            model = resnetf(
                "resnet34",
                classes,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "resnet50":
            model = resnetf(
                "resnet50",
                classes,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif arch == "resnet101":
            model = resnetf(
                "resnet101",
                classes,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif "resnet18Py" in arch:
            model = resnetfPy(
                "resnet18", 
                classes, 
                precision, 
                bit_error_rate, 
                position, 
                faulty_layers,
            )
        elif "resnet50Py" in arch:
            model = resnetfPy(
                "resnet50", 
                classes, 
                precision, 
                bit_error_rate, 
                position, 
                faulty_layers,
            )
        elif "vgg11Py" in arch:
            model = vggfPy(
                "A", 
                in_channels, 
                classes,
                True,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif "vgg16Py" in arch:
            model = vggfPy(
                "D", 
                in_channels, 
                classes,
                True,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        elif "vgg19Py" in arch:
            model = vggfPy(
                "E", in_channels, 
                classes,
                True,
                precision,
                bit_error_rate,
                position,
                faulty_layers,
            )
        else:
            raise NotImplementedError

    # print(model)
    checkpoint_epoch = -1

    if retrain:
        if not os.path.exists(checkpoint_path):
            for x in range(cfg.epochs, -1, -1):
                if os.path.exists(model_path_from_base(checkpoint_path, x)):
                    checkpoint_path = model_path_from_base(checkpoint_path, x)
                    break

        if not os.path.exists(checkpoint_path):
            print("Checkpoint path not exists")
            return model, checkpoint_epoch

        # print("Restoring model from checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])
        # print("restored checkpoint at epoch - ", checkpoint["epoch"])
        # print("Training loss =", checkpoint["loss"])
        # print("Training accuracy =", checkpoint["accuracy"])
        checkpoint_epoch = checkpoint["epoch"]

    return model, checkpoint_epoch


def init_models_pairs(arch, in_channels, precision, retrain, checkpoint_path, faulty_layers, bit_error_rate, position, seed=0, dataset='cifar'):

    """Load the default model as well as the corresponding perturbed model"""

    model, checkpoint_epoch = init_models(
        arch, in_channels, precision, retrain, checkpoint_path, dataset=dataset
    )
    model_p, checkpoint_epoch_p = init_models_faulty(
        arch,
        in_channels,
        precision,
        retrain,
        checkpoint_path,
        faulty_layers,
        bit_error_rate,
        position,
        seed=seed,
        dataset=dataset,
    )

    return model, checkpoint_epoch, model_p, checkpoint_epoch_p


def default_base_model_path(data_dir, arch, dataset, precision, fl, ber, pos):
    extra = [arch, dataset, "p", str(precision), "model"]
    if len(fl) != 0:
        arch = arch + "f"
        extra[0] = arch
        extra.append("fl")
        extra.append("-".join(fl))
        extra.append("ber")
        extra.append("%03.3f" % ber)
        extra.append("pos")
        extra.append(str(pos))
    return os.path.join(data_dir, arch, dataset, "_".join(extra))


def default_model_path(data_dir, arch, dataset, precision, fl, ber, pos, epoch):
    extra = [arch, dataset, "p", str(precision), "model"]
    if len(fl) != 0:
        arch = arch + "f"
        extra[0] = arch
        extra.append("fl")
        extra.append("-".join(fl))
        extra.append("ber")
        extra.append("%03.3f" % ber)
        extra.append("pos")
        extra.append(str(pos))
    extra.append(str(epoch))
    return os.path.join(data_dir, arch, dataset, "_".join(extra) + ".pth")


def model_path_from_base(basename, epoch):
    return basename + "_" + str(epoch) + ".pth"
