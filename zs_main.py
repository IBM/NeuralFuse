import argparse
import os
import sys

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# Sinlge model
import zs_train as train

# EOPM-Based
import zs_train_input_transform_eopm_gen as transform_eopm_gen

# Evaluation
import zs_train_input_transform_eval as transform_eval

from config import cfg
from models import default_base_model_path

np.set_printoptions(threshold=sys.maxsize)
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():

    print("Running command:", str(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arch",
        help="Input network architecture",
        choices=[
            "resnet18",  
            "resnet50", 
            "resnet18Py", 
            "resnet50Py", 
            "vgg11", 
            "vgg16", 
            "vgg19", 
            "vgg11Py", 
            "vgg16Py", 
            "vgg19Py",
        ],
        default="resnet18",
    )
    parser.add_argument(
        "mode",
        help="Specify operation to perform",
        default="train",
        choices=[
            "train", 
            "transform_eval",
            "transform_eopm_gen", 
        ],
    )
    parser.add_argument(
        "dataset",
        help="Specify dataset",
        choices=[
            "cifar10", 
            "cifar100", 
            "gtsrb", 
            "imagenet128", 
            "imagenet224"
        ],
        default="cifar10",
    )
    group = parser.add_argument_group(
        "Reliability/Error control Options",
        "Options to control the fault injection details.",
    )
    group.add_argument(
        "-ber",
        "--bit_error_rate",
        type=float,
        help="Bit error rate for training corresponding to known voltage.",
        default=0.01,
    )
    group.add_argument(
        "-pos",
        "--position",
        type=int,
        help="Position of bit errors.",
        default=-1,
    )
    group = parser.add_argument_group(
        "Initialization options", "Options to control the initial state."
    )
    group.add_argument(
        "-rt",
        "--retrain",
        action="store_true",
        help="Continue training on top of already trained model."
        "It will start the "
        "process from the provided checkpoint.",
        default=False,
    )
    group.add_argument(
        "-cp",
        "--checkpoint",
        help="Name of the stored checkpoint that needs to be "
        "retrained or used for test (only used if -rt flag is set).",
        default=None,
    )
    group.add_argument(
        "-F",
        "--force",
        action="store_true",
        help="Do not fail if checkpoint already exists. Overwrite it.",
        default=False,
    )
    group = parser.add_argument_group(
        "Other options", "Options to control training/validation process."
    )
    group.add_argument(
        "-E",
        "--epochs",
        type=int,
        help="Maxium number of epochs to train.",
        default=5,
    )
    group.add_argument(
        "-LR",
        "--learning_rate",
        type=float,
        help="Learning rate for training input transformation of training clean model.",
        default=5,
    )
    group.add_argument(
        "-LM",
        "--lambdaVal",
        type=float,
        help="Lambda value between two loss function",
        default=1,
    )
    group.add_argument(
        "-BS",
        "--batch-size",
        type=int,
        help="Training batch size.",
        default=128,
    )
    group.add_argument(
        "-TBS",
        "--test-batch-size",
        type=int,
        help="Test batch size.",
        default=100,
    )
    group.add_argument(
        "-N",
        "--N_perturbed_model",
        type=int,
        help="How many perturbed model will be used for training.",
        default=100,
    )
    group.add_argument(
        "-G",
        "--Generator",
        type=str,
        help="Which generator to be used.",
        default='large',
    )

    args = parser.parse_args()
    cfg.epochs = args.epochs
    cfg.learning_rate = args.learning_rate
    cfg.batch_size = args.batch_size
    cfg.test_batch_size = args.test_batch_size
    cfg.lb = args.lambdaVal
    cfg.N = args.N_perturbed_model
    cfg.G = args.Generator

    print("Preparing data..", args.dataset)
    if args.dataset == "cifar10":
        dataset = "cifar10"
        in_channels = 3
        
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )
            
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root=cfg.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.CIFAR10(
            root=cfg.data_dir,
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif args.dataset == "cifar100":
        dataset = "cifar100"
        in_channels = 3
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root=cfg.data_dir,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.CIFAR100(
            root=cfg.data_dir,
            train=False,
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif args.dataset == "gtsrb":
        dataset = "gtsrb"
        in_channels = 3
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomAffine(degrees = 0, translate=(0.35, 0.35), scale=(0.65, 1.35)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.GTSRB(
            root=cfg.data_dir,
            split="train",
            download=True,
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.GTSRB(
            root=cfg.data_dir,
            split='test',
            download=True,
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif args.dataset == "imagenet128":
        dataset = "imagenet128"
        in_channels = 3
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.ImageNet(
            root='data/imagenet-10/',
            split="train",
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.ImageNet(
            root='data/imagenet-10/',
            split="val",
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif args.dataset == "imagenet224":
        dataset = "imagenet224"
        in_channels = 3
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t * 2 - 1),
            ]
        )

        trainset = torchvision.datasets.ImageNet(
            root='data/imagenet-10/',
            split="train",
            transform=transform_train,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=8
        )

        testset = torchvision.datasets.ImageNet(
            root='data/imagenet-10/',
            split="val",
            transform=transform_test,
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=2,
        )

    print("Device", device)
    cfg.device = device

    assert isinstance(cfg.faulty_layers, list)

    if args.checkpoint is None and args.mode != "transform":
        args.checkpoint = default_base_model_path(
            cfg.data_dir,
            args.arch,
            dataset,
            cfg.precision,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.checkpoint is None and args.mode == "transform":
        args.checkpoint = []
        args.checkpoint.append(
            default_base_model_path(
                cfg.data_dir,
                args.arch,
                dataset,
                cfg.precision,
                [],
                args.bit_error_rate,
                args.position,
            )
        )
        args.checkpoint.append(
            default_base_model_path(
                cfg.data_dir,
                args.arch,
                dataset,
                cfg.precision,
                cfg.faulty_layers,
                args.bit_error_rate,
                args.position,
            )
        )

    if args.mode == "train":
        print("training args", args)
        train.training(
            trainloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.retrain,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_eopm_gen":
        print("input_transform_train_eopm_gen", args)
        cfg.save_dir = 'eopm_p_gen/'
        cfg.save_dir_curve = 'eopm_curve_gen/'
        transform_eopm_gen.transform_train(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    elif args.mode == "transform_eval":
        print("input_transform_train_eval", args)
        transform_eval.transform_eval(
            trainloader,
            testloader,
            args.arch,
            dataset,
            in_channels,
            cfg.precision,
            args.checkpoint,
            args.force,
            device,
            cfg.faulty_layers,
            args.bit_error_rate,
            args.position,
        )
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
