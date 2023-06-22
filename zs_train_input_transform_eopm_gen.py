from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch

from collections import OrderedDict
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm
import copy

from models import init_models_pairs, create_faults
from models.generator import *
import faultsMap as fmap
from config import cfg

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

def compute_loss(model_outputs, labels):
    _, preds = torch.max(model_outputs, 1)
    labels = labels.view(labels.size(0))  # changing the size from (batch_size,1) to batch_size.
    loss = nn.CrossEntropyLoss()(model_outputs, labels)
    return loss, preds

def accuracy_checking(model_orig, model_p, trainloader, testloader, gen, device, use_transform=False):
    """
    Check the accuracy for both training data and testing data.
    :param model_orig: The clean model.
    :param model_p: The perturbed model.
    :param trainloader: The loader of training data.
    :param testloader: The loader of testing data.
    :param gen: Generator object to generate the perturbation based on the input images.
    :param device: Specify GPU usage.
    :param use_transform: Whether to apply input transfomation or not.
    """
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
            x_adv = gen(x)
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
            x_adv = gen(x)
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
            accuracy_orig_train, accuracy_p_train
        )
    )
    print("Accuracy of testing data: clean model: {:5f}, perturbed model: {:5f}".format(
            accuracy_orig_test, accuracy_p_test
        )
    )

    return accuracy_orig_train, accuracy_p_train, accuracy_orig_test, accuracy_p_test
 
def init_dict(model_transform, grad_dict):
    for param_name, param_weight in model_transform.named_parameters():
        if param_weight.requires_grad:
            grad_dict[param_name] = 0

def sum_grads(model_transform, grad_dict):
    for param_name, param_weight in model_transform.named_parameters():
        if param_weight.requires_grad:
            grad_dict[param_name] += param_weight.grad 

def mean_grads(grad_dict, nums):
    param_names = grad_dict.keys()
    for param_name in param_names:
        grad_dict[param_name] /= nums

def transform_train(trainloader, testloader, arch, dataset, in_channels, precision, checkpoint_path, force, device, fl, ber, pos, seed=0):
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

    storeLoss = []

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
        
    Gen = Gen.to(device)

    # Using Adam:
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, Gen.parameters()),
        lr=cfg.learning_rate,
        betas=(0.5, 0.999),
        # weight_decay=5e-4,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=cfg.decay)
    lb = cfg.lb  # Lambda 

    for name, param in Gen.named_parameters():
        print("Param name: {}, grads is: {}".format(
            name, 
            param.requires_grad
            )
        )

    print('========== Check setting: Epoch: {}, Batch_size: {}, N perturbed models: {}, Lambda: {}, BitErrorRate: {}, LR: {}, G: {}, Random Training: {}=========='.format(
        cfg.epochs, 
        cfg.batch_size, 
        cfg.N, 
        lb, 
        ber, 
        cfg.learning_rate, 
        cfg.G, 
        cfg.totalRandom
        )
    )

    print("========== Start training the parameter of the input transform by using EOT attack ==========")

    # Initialization clean and perturbed model
    create_faults(precision, ber, pos, seed=0)
    model, _, model_perturbed, _ = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl, ber, pos, dataset=dataset)
    model, model_perturbed =  model.to(device), model_perturbed.to(device)


    if 'Py' in arch: # For Larger Image Size
        summary(Gen, (3, 224, 224))
        summary(model, (3, 224, 224))
    elif 'imagenet128' in dataset:
        summary(Gen, (3, 128, 128))
        summary(model, (3, 128, 128))
    elif 'imagenet224' in dataset:
        summary(Gen, (3, 224, 224))
        summary(model, (3, 224, 224))
    else: 
        summary(Gen, (3, 32, 32))
        summary(model, (3, 32, 32))

    for epoch in range(cfg.epochs):
        running_loss = 0
        running_correct_orig = 0
        running_correct_p = 0
        each_c_pred = [0] * cfg.N
        each_p_pred = [0] * cfg.N
              
        # For each epoch, we will use N perturbed model for training.
        for batch_id, (image, label) in tqdm.tqdm(enumerate(trainloader)):
            gradDict = OrderedDict()
            init_dict(Gen, gradDict)

            image, label = image.to(device), label.to(device)
            
            for k in range(cfg.N):
                
                loss = 0

                image_adv = Gen(image) 
                image_adv = image_adv.to(device)
                
                # Random test
                if cfg.totalRandom:
                    j = random.randint(0, cfg.randomRange)
                else:
                    j = k

                # Reset the BitErrorMap for different perturbed models.
                fmap.BitErrorMap0to1 = None 
                fmap.BitErrorMap1to0 = None
                
                # Create faultMap for faultinjection_ops.
                create_faults(precision, ber, pos, seed=j)                  
                
                model.eval()
                model_perturbed.eval()
                Gen.train()

                # Inference the clean model and perturbed model
                out_biterror_without_p = model_perturbed(image) 
                _, pred_without_p = torch.max(out_biterror_without_p, 1)
                each_c_pred[k] += torch.sum(pred_without_p == label.data).item()

                out = model(image_adv)  # pylint: disable=E1102
                out_biterror = model_perturbed(image_adv)  # pylint: disable=E1102   
                              
                # Compute the loss for clean model and perturbed model
                loss_orig, pred_orig = compute_loss(out, label)
                loss_p, pred_p = compute_loss(out_biterror, label)     

                each_p_pred[k] += torch.sum(pred_p == label.data).item()     
                    
                # Keep the running accuracy of clean model and perturbed model.
                running_correct_orig += torch.sum(pred_orig == label.data).item()
                running_correct_p += torch.sum(pred_p == label.data).item() 

                loss = loss_orig + lb * loss_p

                # Keep the overal loss for whole batches
                running_loss += loss.item()  

                # Calculate the gradients
                optimizer.zero_grad()
                loss.backward()
                
                # Sum all of the gradients
                sum_grads(Gen, gradDict)
    
            # Average the gradients
            mean_grads(gradDict, cfg.N)

            # Set gradients back to P
            for param_name, param_weight in Gen.named_parameters():
                param_weight.grad = gradDict[param_name]

            # Apply gradients by optimizer to parameter           
            optimizer.step()
            lr_scheduler.step()
            
        print('Each pred w/o transformation: {}'.format(
                [np.round(x/len(trainloader.dataset), decimals=4) for x in each_c_pred]
            )
        )
        print('Each pred with transformation: {}'.format(
                [np.round(x/len(trainloader.dataset), decimals=4) for x in each_p_pred]
            )
        )
            
        # Keep the running accuracy of clean model and perturbed model for all mini-batch.
        accuracy_orig = running_correct_orig / (len(trainloader.dataset) * cfg.N)
        accuracy_p = running_correct_p / (len(trainloader.dataset) * cfg.N)
        print("For epoch: {}, loss: {:.6f}, accuracy for {} clean model: {:.5f}, accuracy for {} perturbed model: {:.5f}".format(
                epoch + 1,
                running_loss / cfg.N,
                cfg.N,
                accuracy_orig,
                cfg.N,
                accuracy_p,
            )
        )

        storeLoss.append(running_loss / cfg.N)

        if (epoch + 1) % 50 == 0 or (epoch + 1) == cfg.epochs:
            # Saving the result of the generator!
            torch.save(Gen.state_dict(),
                cfg.save_dir + 'EOPM_Generator{}Q_{}_arch_{}_LR{}_E_{}_ber_{}_lb_{}_N_{}_step500_NOWE_{}.pt'.format(
                    cfg.G, 
                    dataset, 
                    arch, 
                    cfg.learning_rate, 
                    cfg.epochs, 
                    ber, 
                    lb, 
                    cfg.N, 
                    epoch+1
                )
            )

    print('========== Start checking the accuracy with different perturbed model ==========')
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

    model, _, model_perturbed, _ = init_models_pairs(arch, in_channels, precision, True, checkpoint_path, fl, ber, pos, dataset=dataset)
    model, model_perturbed = model.to(device), model_perturbed.to(device)
    for i in range(cfg.beginSeed, cfg.endSeed):
        print(' ********** For seed: {} ********** '.format(i))
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
        np.std(accuracy_p_test_list_with_transformation))
    )
