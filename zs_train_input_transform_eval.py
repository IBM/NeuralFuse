import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from config import cfg

from models import vggfPy 
from models import resnetfPy
from models import init_models_pairs, create_faults, init_models
from models.generator import *
import faultsMap as fmap

from collections import OrderedDict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import itertools
import numpy as np
import tqdm
import copy


torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-20
PGD_STEP = 2


def accuracy_checking_clean(model_orig, trainloader, testloader, device):
    """
      Calculating the accuracy with given clean model and perturbed model.
      :param model_orig: Clean model.
      :param model_p: Perturbed model.
      :param trainloader: The loader of training data.
      :param testloader: The loader of testing data.
      :param transform_model: The object of transformation model.
      :param device: Specify GPU usage.
      :use_transform: Should apply input transformation or not.
    """
    cfg.replaceWeight = False
    total_train = 0
    total_test = 0
    correct_orig_train = 0
    correct_p_train = 0
    correct_orig_test = 0
    correct_p_test = 0

    # For training data:
    for x, y in trainloader:
        total_train += 1
        x, y = x.to(device), y.to(device)
        out_orig = model_orig(x)
        _, pred_orig = out_orig.max(1)
        y = y.view(y.size(0))
        correct_orig_train += torch.sum(pred_orig == y.data).item()
    accuracy_orig_train = correct_orig_train / (len(trainloader.dataset))

    # For testing data:
    for x, y in testloader:
        total_test += 1
        x, y = x.to(device), y.to(device)
        out_orig = model_orig(x)
        _, pred_orig = out_orig.max(1)
        y = y.view(y.size(0))
        correct_orig_test += torch.sum(pred_orig == y.data).item()
    accuracy_orig_test = correct_orig_test / (len(testloader.dataset))

    print("Accuracy of training data: clean model: {:5f}".format(accuracy_orig_train))
    print("Accuracy of testing data: clean model: {:5f}".format(accuracy_orig_test))


def accuracy_checking(model_orig, model_p, trainloader, testloader, transform_model, device, use_transform=False):
    """
      Calculating the accuracy with given clean model and perturbed model.
      :param model_orig: Clean model.
      :param model_p: Perturbed model.
      :param trainloader: The loader of training data.
      :param testloader: The loader of testing data.
      :param transform_model: The object of transformation model.
      :param device: Specify GPU usage.
      :use_transform: Should apply input transformation or not.
    """
    cfg.replaceWeight = False
    total_train = 0
    total_test = 0
    correct_orig_train = 0
    correct_p_train = 0
    correct_orig_test = 0
    correct_p_test = 0

    # For training data:
    for x, y in trainloader:
        total_train += 1
        x, y = x.to(device), y.to(device)
        if use_transform:
            x_adv = transform_model(x)
            out_orig = model_orig(x_adv)
            out_p = model_p(x_adv)
        else:
            out_orig = model_orig(x)
            out_p = model_p(x)
        _, pred_orig = out_orig.max(1)
        _, pred_p = out_p.max(1)
        y = y.view(y.size(0))
        correct_orig_train += torch.sum(pred_orig == y.data).item()
        correct_p_train += torch.sum(pred_p == y.data).item()
    accuracy_orig_train = correct_orig_train / (len(trainloader.dataset))
    accuracy_p_train = correct_p_train / (len(trainloader.dataset))

    # For testing data:
    for x, y in testloader:
        total_test += 1
        x, y = x.to(device), y.to(device)
        if use_transform:
            x_adv = transform_model(x)
            out_orig = model_orig(x_adv)
            out_p = model_p(x_adv)
        else:
            out_orig = model_orig(x)
            out_p = model_p(x)
        _, pred_orig = out_orig.max(1)
        _, pred_p = out_p.max(1)
        y = y.view(y.size(0))
        correct_orig_test += torch.sum(pred_orig == y.data).item()
        correct_p_test += torch.sum(pred_p == y.data).item()
    accuracy_orig_test = correct_orig_test / (len(testloader.dataset))
    accuracy_p_test = correct_p_test / (len(testloader.dataset))

    print("Accuracy of training data: clean model: {:5f}, perturbed model: {:5f}".format(
            accuracy_orig_train, 
            accuracy_p_train
        )
    )
    print("Accuracy of testing data: clean model: {:5f}, perturbed model: {:5f}".format(
            accuracy_orig_test, 
            accuracy_p_test
        )
    )

    return accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test

def transform_eval(
    trainloader,
    testloader,
    arch,
    dataset,
    in_channels,
    precision,
    checkpoint_path,
    force,
    device,
    fl,
    ber,
    pos,
):
    """
    Apply quantization aware training.
    :param trainloader: The loader of training data.
    :param in_channels: An int. The input channels of the training data.
    :param arch: A string. The architecture of the model would be used.
    :param dataset: A string. The name of the training data.
    :param ber: A float. How many rate of bits would be attacked.
    :param precision: An int. The number of bits would be used to quantize
                      the model.
    :param position:
    :param checkpoint_path: A string. The path that stores the models.
    :param device: Specify GPU usage.
    """
    torch.backends.cudnn.benchmark = True

    if(cfg.testing_mode == 'clean'):

        model, checkpoint_epoch = init_models(arch, 3, precision, True, checkpoint_path, dataset)

        model = model.to(device)
        model.eval()
        accuracy_checking_clean(model, trainloader, testloader, device)

    if cfg.testing_mode == 'generator_base':
        
        if cfg.G == 'ConvL':
            Gen = GeneratorConvLQ(precision)
        elif cfg.G == 'ConvS':
            Gen = GeneratorConvSQ(precision)
        elif cfg.G == 'DeConvL':
            Gen = GeneratorDeConvLQ(precision)
        elif cfg.G == 'DeConvS':
            Gen = GeneratorDeConvSQ(precision)
        elif cfg.G == 'UNetL':
            Gen = GeneratorUNetLQ(precision)
        elif cfg.G == 'UNetS':
            Gen = GeneratorUNetSQ(precision)
        
        Gen.load_state_dict(torch.load(cfg.G_PATH))
        Gen = Gen.to(device)
        print('Successfully loading the generator model.')

        print('========== Start checking the accuracy with different perturbed model: bit error mode ==========')
        # Setting without input transformation
        accuracy_orig_train_list = []
        accuracy_p_train_list = []
        accuracy_orig_test_list = []
        accuracy_p_test_list = []
    
        # Setting with input transformation
        accuracy_orig_train_list_with_transformation = []
        accuracy_p_train_list_with_transformation = []
        accuracy_orig_test_list_with_transformation = []
        accuracy_p_test_list_with_transformation = []
    
        for i in range(50000, 50010):
            print(' ********** For seed: {} ********** '.format(i))
            (model, checkpoint_epoch, model_perturbed, checkpoint_epoch_perturbed) = init_models_pairs( 
                        arch, in_channels, precision, True, checkpoint_path, fl,  ber, pos, seed=i, dataset=dataset)
            model, model_perturbed = model.to(device), model_perturbed.to(device),
            fmap.BitErrorMap0to1 = None 
            fmap.BitErrorMap1to0 = None
            create_faults(precision, ber, pos, seed=i)
            model.eval()
            model_perturbed.eval()
            Gen.eval()
    
            # Without using transform
            accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=False)
            accuracy_orig_train_list.append(accuracy_orig_train)
            accuracy_p_train_list.append(accuracy_p_train)
            accuracy_orig_test_list.append(accuracy_orig_test)
            accuracy_p_test_list.append(accuracy_p_test)
    
            # With input transform
            accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test = accuracy_checking(model, model_perturbed, trainloader, testloader, Gen, device, use_transform=True)
            accuracy_orig_train_list_with_transformation.append(accuracy_orig_train)
            accuracy_p_train_list_with_transformation.append(accuracy_p_train)
            accuracy_orig_test_list_with_transformation.append(accuracy_orig_test)
            accuracy_p_test_list_with_transformation.append(accuracy_p_test)


        # Without using transform
        print('The average results without input transformation -> accuracy_orig_train: {:5f}, accuracy_p_train: {:5f}, accuracy_orig_test: {:5f}, accuracy_p_test: {:5f}'.format(
                np.mean(accuracy_orig_train_list), 
                np.mean(accuracy_p_train_list), 
                np.mean(accuracy_orig_test_list), 
                np.mean(accuracy_p_test_list)
            )
        )
        print('The average results without input transformation -> std_accuracy_orig_train: {:5f}, std_accuracy_p_train: {:5f}, std_accuracy_orig_test: {:5f}, std_accuracy_p_test: {:5f}'.format(
                np.std(accuracy_orig_train_list), 
                np.std(accuracy_p_train_list), 
                np.std(accuracy_orig_test_list), 
                np.std(accuracy_p_test_list)
            )
        )
    
        print()
    
        # With input transform
        print('The average results with input transformation -> accuracy_orig_train: {:5f}, accuracy_p_train: {:5f}, accuracy_orig_test: {:5f}, accuracy_p_test: {:5f}'.format(
                np.mean(accuracy_orig_train_list_with_transformation), 
                np.mean(accuracy_p_train_list_with_transformation), 
                np.mean(accuracy_orig_test_list_with_transformation), 
                np.mean(accuracy_p_test_list_with_transformation)
            )
        )
        print('The average results with input transformation -> std_accuracy_orig_train: {:5f}, std_accuracy_p_train: {:5f}, std_accuracy_orig_test: {:5f}, std_accuracy_p_test: {:5f}'.format(
                np.std(accuracy_orig_train_list_with_transformation), 
                np.std(accuracy_p_train_list_with_transformation), 
                np.std(accuracy_orig_test_list_with_transformation), 
                np.std(accuracy_p_test_list_with_transformation)
            )
        ) 