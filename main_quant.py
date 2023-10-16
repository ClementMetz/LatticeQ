import math
import argparse
import os
import copy
from datetime import date
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models

import sys

from torchTransformer import TorchTransformer
import quantize_filters
from QConv2d import QConv2d
from QLinear import QLinear
from QPooling import QMaxPool2d,QAdaptiveAvgPool2d
from torchvision.models.resnet import BasicBlock

from tqdm.autonotebook import tqdm

from util import AverageMeter, accuracy, adjust_learning_rate, optimizer, forward_activation, EndForward



def parse_option():
    parser = argparse.ArgumentParser('argument for calibrating')

    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--bitsW', default=8, type=int, metavar='NBW', help='bitwidth for weights')
    parser.add_argument('--bitsA', default=32, type=int, metavar='NBA', help='bitwidth for activations')
    parser.add_argument('--quant_a', action='store_true', help='enable activation quantization')
    parser.add_argument('--quant_a_type', default='per_layer', choices=['per_layer','per_channel'], help='type of activation quantization (naive per layer or quantile per channel)')
    parser.add_argument('--agreg',default=2, type=int, help='quantization dimension for 1x1 convolution layers and linear layers')
    parser.add_argument('--traindir', type=str, default='', help='training dataset directory to get calibration sample, default is valdir')
    parser.add_argument('--valdir', type=str, help='validation dataset directory to evaluate the network')
    parser.add_argument('--bias_corr',action = 'store_true', help='whether or not use bias correction for quantization')
    parser.add_argument('--per_layer', action = 'store_true', help='True if you want to quantize per layer instead of per channel')
    parser.add_argument('--quant_basis',action = 'store_true', help='quantize bases')
    parser.add_argument('--bitsB',type=int,default=8,help='Number of bits for bases')
    parser.add_argument('--bit_alloc_w',action = 'store_true', help = 'allow per channel bit allocation for weights')
    parser.add_argument('--bit_alloc_a',action = 'store_true', help = 'allow per channel bit allocation for activations')
    parser.add_argument('--nb_restarts',type=int,default=5, help = 'number of restarts for random search')
    parser.add_argument('--steps', type=int, default=80, help='number of steps for random search')
    parser.add_argument('--random_seed',type=int,default=1, help = 'random seed for experiment')
    parser.add_argument('--block_reconstruction',action='store_true',help='activate if you want to perform block reconstruction')
    parser.add_argument('--epochs',type=int,default=2000, help='number of epochs for block reconstruction')
    parser.add_argument('--cali_batches',type=int,default=4, help='number of batch slices for block reconstruction')
    parser.add_argument('--lr',type=float,default=2.0*10**-6, help='learning rate for block reconstruction')
    parser.add_argument('--alpha_lr',type=float,default=1.0*10**-3, help='learning rate for alpha for block reconstruction')
    opt = parser.parse_args()
    if opt.traindir=='':
        opt.traindir = opt.valdir
    opt.model_name = "models."+opt.model_name
    torch.manual_seed(opt.random_seed)
    opt.conv_idx=0
    opt.conv_idx2=0
    opt.nb_linear = 0
    opt.counted_linear = 0
    opt.handles=[]
    opt.activation = {}
    opt.FP_activation = {}
    today = date.today()

    log = ' top-1 Accuracy '
    log += ' LatticeQ '
    log += 'conv(1,1)->' + str(opt.bitsW) + 'bits  '
    log += 'conv(3,3)->' + str(opt.bitsW) + 'bits  '
    if opt.quant_a:
        log += ' & quant scalar activations {}bits'.format(opt.bitsA)
    log += ' : '
    opt.log = log
    return opt

def data(opt):
    valdir = opt.valdir
    traindir = opt.traindir

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_data = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    

    # DATA_loader
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=opt.batch_size,
                                                shuffle=True, 
                                                num_workers=opt.num_workers, 
                                                pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=opt.batch_size,#opt.batch_size, 
                                                shuffle=True, 
                                                num_workers=opt.num_workers,
                                                pin_memory=True)

    return train_loader, val_loader




def validate(val_loader, model, criterion):
    """validation"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            if (idx +1) % 10 == 0:
                tqdm.write('Test: [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx+1, len(val_loader), loss=losses, top1=top1))

    tqdm.write(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    

    return losses.avg, top1.avg


def block_reconstruction(opt,current_layer,model,FP_model,images,parent_name=''):
    for module_name in current_layer._modules:
        m = current_layer._modules[module_name]
        if type(m) in [BasicBlock,QConv2d,QLinear] or ("BasicBlock" in str(type(m))): #ugly, works only for resnet, add your blocktype to allow for blockwise reconstruction of other architectures
            
            name = parent_name+'_'+module_name
            print(name)
            optim = optimizer(m,opt)
            criterion = nn.MSELoss().cuda()

            init_hook=True
            x = images
            n = images.shape[0]
            bs = n//opt.cali_batches

            for j in range(opt.cali_batches):
                x0 = x[j*bs:(j+1)*bs].float().cuda()
                yb = forward_activation(opt, FP_model, x0, name, init_hook=init_hook)
                init_hook=False
                opt.FP_activation[name+'_'+str(j)] = opt.activation[name+'_output'] #transfer output to FP_activation
                
            opt.activation.clear()
            init_hook=True
            ep = opt.epochs
            for k in range(ep):
                new_lr = adjust_learning_rate(opt, optim, k, ep)

                optim.zero_grad()
                
                for j in range(opt.cali_batches):
                    x0 = x[j*bs:(j+1)*bs].float().cuda()
                    yb = forward_activation(opt, model, x0, name, init_hook=init_hook)
                    init_hook = False

                    loss = criterion(opt.activation[name+'_output'],opt.FP_activation[name+'_'+str(j)].clone().detach())
                    loss.backward()
                
                    optim.step()
                
                if k%100==0 or k==ep-1:
                    print('['+str(k)+']: '+'MSE = ' + str(loss.item()))
                    #print(m.alpha.item())
                    
            opt.FP_activation.clear()
            opt.activation.clear()    

            for h in opt.handles: #remove all hooks
                h.remove()
            


        elif len(m._modules) > 0:
            block_reconstruction(opt, m, model, FP_model, images, parent_name+'_'+module_name)




def Calibrate_filters(opt,model,parent_name=''): #Goes through the network and sets up every layer for quantization
    for module_name in model._modules:
        s = opt.steps
        m = model._modules[module_name]

        if type(m) == QConv2d:
            log = parent_name+'_'+module_name + '_kernel_size = ' + str(m.kernel_size)
            m.quant_w = True
            m.quant_a = opt.quant_a
            m.quant_basis=opt.quant_basis
            m.per_layer = opt.per_layer
            m.bitsb = opt.bitsB
            m.bias_corr = opt.bias_corr

            if opt.conv_idx==0: #first layer
                m.quant_a = False
                m.set_bits(8,8)
                m.set_agregation(1)
                sc = 1/((2**(m.bitsw))-1) #scale for random search

                if m.per_layer:
                    m.calibrate_base_per_layer(steps=[sc/10**4,sc,sc/4,sc/9],step_lengths=[100,100,100,100],init_basis=True,learn=False)
                else:
                    m.calibrate_base_per_channel(steps=[sc/10**4,sc,sc/4,sc/9],step_lengths=[100,100,100,100],init_basis=True,learn=False)
                stri = 'per layer' if opt.per_layer else 'per channel'
                
                log += ' ----->      LatticeQ weights {}bits activations {}bits agreg{} weights {} activations {}'.format(m.bitsw, m.bitsa, m.agreg, stri, m.quant_a_type)
                
            else: #other conv layers
                m.bit_alloc_a = opt.bit_alloc_a
                m.bit_alloc_w = opt.bit_alloc_w
                m.quant_a_type=opt.quant_a_type
                bitsW = opt.bitsW

                #Quantiles allocation
                if "mobilenet" in opt.model_name:
                    m.d = {0:0,3:0.986,4:0.998,8:1}
                elif "vgg" in opt.model_name:
                    m.d = {0:0,3:0.9999,4:0.9999,8:1}
                else:
                    m.d = {0:0,1:0.97,2:0.992,3:0.9991,4:0.9997,5:0.9998,6:0.99995,7:0.99999,8:1}
                
                #Basis dimension setting
                if m.kernel_size == (1,1):
                    agreg = opt.agreg
                
                elif m.kernel_size == (3,3):
                    agreg = 3
                        
                else:
                    raise ValueError(' Unexpected kernel size {}'.format(m.kernel_size))

                m.set_bits(bitsW, opt.bitsA)
                m.set_agregation(agreg)

                sc = 1/((2**(m.bitsw))-1) #scale for random search

                if m.per_layer: #per layer quantization
                    if m.agreg==1: #scalar quantization
                        for b in range(opt.nb_restarts):
                            m.calibrate_base_per_layer(steps=[sc/10**4,sc,sc/4,sc/9],step_lengths=[100,100,100,100],init_basis=(b==0),learn=False)
                    else:
                        for b in range(opt.nb_restarts):
                            m.calibrate_base_per_layer(steps=[sc/10**4,sc,sc/2,sc/3,sc/5,sc/7,sc/9,sc/15,sc/30],step_lengths=[s,s,s,s,s,s,s,s,s],init_basis=(b==0),learn=False) #naive pre calibration
                        
                        
                else: #per channel quantization
                    if m.agreg==1: #scalar quantization
                        for b in range(opt.nb_restarts):
                            m.calibrate_base_per_channel(steps=[sc/10**4,sc,sc/4,sc/9],step_lengths=[100,100,100,100],init_basis=(b==0),learn=False)
                    else:
                        for b in range(opt.nb_restarts):
                            m.calibrate_base_per_channel(steps=[sc/10**4,sc,sc/2,sc/3,sc/5,sc/7,sc/9,sc/15,sc/30],step_lengths=[s,s,s,s,s,s,s,s,s],init_basis=(b==0),learn=False) #naive pre calibration
                stri = 'per layer' if opt.per_layer else 'per channel'
                
                log += ' ----->      LatticeQ weights {}bits activations {}bits agreg{} weights {} activations {}'.format(m.bitsw, m.bitsa, m.agreg, stri, m.quant_a_type)

            print(log)
            
            opt.conv_idx += 1

        elif type(m) == QLinear: #linear layers
            opt.counted_linear += 1
            m = model._modules[module_name]
            if opt.conv_idx > 0:
                log = parent_name+'_'+module_name
                m.quant_w=True
                m.quant_a=opt.quant_a
                m.quant_basis=opt.quant_basis
                agreg = opt.agreg

                if opt.counted_linear == opt.nb_linear: #last layer

                    bitsW = 8
                    if opt.quant_a:
                        bitsA=8
                    else:
                        bitsA=32
                    m.bitsb = 8
                    m.per_layer = True
                    m.bit_alloc_w = False
                    m.bit_alloc_a = False
                    m.bias_corr = False

                else: #VGG has several fully connected layers
                    
                    bitsW = opt.bitsW
                    bitsA = opt.bitsA
                    m.bitsb = opt.bitsB
                    m.quant_a_type = opt.quant_a_type
                    m.per_layer = opt.per_layer
                    m.bit_alloc_w = opt.bit_alloc_w
                    m.bit_alloc_a = opt.bit_alloc_a
                    m.bias_corr = opt.bias_corr

                    if "vgg" in opt.model_name:
                        m.d = {0:0,3:0.9999,4:0.9999,8:1}
                        
                    else:
                        m.d = {0:0,1:0.97,2:0.992,3:0.9991,4:0.9997,5:0.9998,6:0.99995,7:0.99999,8:1} #0.9991 #0.9997
                
                m.set_bits(bitsW, bitsA)
                m.set_agregation(agreg)
                
                if m.per_layer:
                    sc = 1/((2**(m.bitsw))-1)

                    for b in range(1):
                        m.calibrate_base_per_layer(steps=[sc/10**4,sc,sc/2,sc/3,sc/5,sc/7,sc/9,sc/15,sc/30],step_lengths=[300,300,300,300,300,300,300,300,300],init_basis=(b==0),learn=False) #naive pre calibration
                
                else: #per channel fc layers for VGG
                    sc = 1/((2**(m.bitsw))-1)

                    for b in range(1):
                        m.calibrate_base_per_channel(steps=[sc/10**4,sc,sc/2,sc/3,sc/5,sc/7,sc/9,sc/15,sc/30],step_lengths=[300,300,300,300,300,300,300,300,300],init_basis=(b==0),learn=False) #naive pre calibration
                stri = 'per layer' if opt.per_layer else 'per channel'

                log += ' ----->      LatticeQ weights {}bits activations {}bits agreg{} weights {} activations {}'.format(m.bitsw, m.bitsa, m.agreg, stri, m.quant_a_type)
                
                print(log)
        
        #Pooling layers
        elif type(m) == QAdaptiveAvgPool2d:
            m = model._modules[module_name]
            if opt.quant_a:
                m.quant_a = True
                m.set_bits(8)
        
        elif type(m) == QMaxPool2d:
            m = model._modules[module_name]
            if opt.quant_a:
                m.quant_a = True
                m.set_bits(8)
        
        elif len(m._modules) > 0:
            Calibrate_filters(opt, model._modules[module_name], parent_name+'_'+module_name)


def forward_batch(model,images): #Function to forward a batch of application images in the network
    with torch.no_grad():
        model(images.cuda(non_blocking=True))


def main():

    opt = parse_option()
    
    print('-----> Import model : ' + opt.model_name)
    if opt.model_name=='models.preresnet18':
        model = torch.hub.load('yhhhli/BRECQ', model='resnet18', pretrained=True) #see Brecq's git to access their model
    else:
        model=eval(opt.model_name)(pretrained=True)


    print('-----> Prepare data loaders')
    train_loader, test_loader = data(opt=opt)



    print('-----> Register model with quant overhead')
    #transform the model to a quantized model for quantization
 
    transformer = TorchTransformer()
    transformer.register(nn.Conv2d, QConv2d)
    transformer.register(nn.Linear, QLinear)
    transformer.register(nn.AdaptiveAvgPool2d, QAdaptiveAvgPool2d)
    transformer.register(nn.MaxPool2d, QMaxPool2d)
    
    model = transformer.trans_layers(model)
    FP_model = copy.deepcopy(model)
    #print(model)
    
    print('-----> Load model on GPU') #Multi GPU not supported
    if torch.cuda.is_available():
        model = model.cuda()
        FP_model = FP_model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    

    #torch.backends.cudnn.benchmark = False
    """
    print('-----> Full precision score')
    loss_Q, acc_Q = validate(val_loader=test_loader, model=model, criterion=criterion)
    print('Full Precision top1 accuracy : ' , acc_Q)
    """
    
    for idx,m in enumerate(model.modules()): #count linear layers in case VGG
        if type(m) == QLinear:
            opt.nb_linear += 1

    print('-----> LatticeQ model search') #Model quantization
    Calibrate_filters(opt=opt, model=model)


    print('-----> Calibration batch selection')
    it = iter(train_loader)
    opt.calibration_images,_ = next(it)

    print('-----> Activation calibration for LatticeQ')
    forward_batch(model,opt.calibration_images) #Forward to initialize alphas

    print('-----> Calibration batches selection')
    it = iter(train_loader)
    opt.calibration_images,_ = next(it)

    print('-----> Activation calibration for LatticeQ')
    model.eval()
    forward_batch(model,opt.calibration_images) #forward to initialize alphas

    if opt.block_reconstruction:
        
        print('-----> LatticeQ model evaluation')
        loss_Q1, acc_Q1 = validate(val_loader=test_loader, model=model, criterion=criterion)
        print('LatticeQ top1 accuracy : ' , acc_Q1.item())
        
        print('-----> Block reconstruction')
        model.train() #switch batchnorms and dropout to train mode
        block_reconstruction(opt=opt, current_layer=model, model=model, FP_model=FP_model, images=opt.calibration_images)
            
        """
        model.eval()
        initialize_alphas(model)

        forward_batch(model,opt.calibration_images)
        """

        print('-----> Model evaluation')
        loss_Q2, acc_Q2 = validate(val_loader=test_loader, model=model, criterion=criterion)
        print('LatticeQ top1 accuracy : ' , acc_Q1.item())
        print('LatticeQ with block reconstruction top1 accuracy : ' , acc_Q2.item())
    else:
        print('-----> LatticeQ model evaluation')
        loss_Q1, acc_Q1 = validate(val_loader=test_loader, model=model, criterion=criterion)
        print('LatticeQ top1 accuracy : ' , acc_Q1.item())
  

    print('-----> LatticeQ model evaluation')
    loss_Q1, acc_Q1 = validate(val_loader=test_loader, model=model, criterion=criterion)
    print('LatticeQ top1 accuracy : ' , acc_Q1.item())
  
    

if __name__ == '__main__':
    main()
