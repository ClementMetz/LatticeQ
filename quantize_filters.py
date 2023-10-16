
#File for quantization functions

import torch
from torch.autograd.function import Function
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
from math import sqrt

"""
    Quantization
"""
class q_filter(Function):
    """
        This is the quantization module.
        The input and output should be all on the interval [0, 1].
        bit is only defined on positive integer values.
    """
    @staticmethod
    def forward(ctx, input):
        res = torch.round(input)
        return res
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def quantize_basis_channel(B,bitwidth): #Function to quantize basis in per-channel setting
    threshold = torch.max(torch.abs(B),dim=1,keepdim=True)[0].max(dim=2,keepdim=True)[0]
    B = (B/threshold + torch.ones_like(B))/2 #Bring all values between 0 and 1
    B = torch.round((2**bitwidth - 1)*B)/(2**bitwidth - 1)
    B = (2*B - torch.ones_like(B))*threshold
    return(B)

def quantize_basis_layer(B,bitwidth): #Function to quantize basis in per-layer setting
    threshold = torch.max(torch.abs(B))
    B = (B/threshold + torch.ones_like(B))/2 #Bring all values between 0 and 1
    B = torch.round((2**bitwidth - 1)*B)/(2**bitwidth - 1)
    B = (2*B - torch.ones_like(B))*threshold
    return(B)

def gram_schmidt_torch_channel(B): #Gram-Schmidt algorithm in per-channel setting
    Btilde = B
    n = Btilde.shape[1]
    for i in range(n-1):
        Bi = Btilde.narrow(1,i,1)
        m = Btilde.matmul(Bi.transpose(1,2))
        denom = m.narrow(1,i,1).detach()

        index = torch.tensor([j for j in range(i+1)]).cuda()
        mask = torch.ones_like(m, dtype=torch.float).cuda()
        mask.index_fill_(1,index,0.0)

        numerator = m * mask
        factor = numerator / denom
        factor = factor.expand(-1,-1,n)
        Bis = Bi.expand(-1,n,-1) 
        Btilde = Btilde - factor * Bis

    return Btilde


def gram_schmidt_torch(B): #Gram-Schmidt algorithm in per-layer setting
    Btilde = B
    n = Btilde.shape[0]
    for i in range(n-1):
        Bi = Btilde.narrow(0,i,1)
        m = Btilde.matmul(Bi.t())
        denom = m.narrow(0,i,1).detach()

        index = torch.tensor([j for j in range(i+1)]).cuda()
        mask = torch.ones_like(m, dtype=torch.float).cuda()
        mask.index_fill_(0,index,0.0)

        numerator = m * mask
        factor = numerator / denom
        factor = factor.expand(-1,n)
        Bis = Bi.expand(n,-1) 
        Btilde = Btilde - factor * Bis

    return Btilde



def babai_torch_channel(bitsw, B, Btilde, v): #Babai algorithm in per-channel setting
    #bitsw is per-channel bitwidths vector, B is bases vector, Btilde is Gram-Schmidt bases vector, v is tensor to quantize
    n = B.shape[1]
    s = -v
    norm = torch.einsum('abs,abs->ab', Btilde, Btilde)
    
    c = torch.zeros_like(v)
    a = torch.floor(2**(bitsw-1))
    a = a.unsqueeze(1).unsqueeze(2).cuda()
    for i in range(n):
        B_vec = B.narrow(1,n-i-1,1)

        tmp = torch.matmul(s,Btilde.transpose(1,2))

        index = torch.tensor(n-i-1).cuda()
        mask = torch.zeros_like(tmp, dtype=torch.float).cuda()

        mask.index_fill_(-1,index,1.0)

        tmp = tmp * mask

        tmp = tmp / norm.narrow(1,n-i-1,1).reshape(-1,1,1)

        tmp = q_filter.apply(tmp)
        
        s = s - tmp.sum(-1, keepdim=True)*B_vec
 
        tmp = torch.max(-a,torch.min(tmp,a-1)) #for bit alloc, equivalent to tmp = torch.clamp(tmp, -a, a-1)
        
        c = c - tmp

    return c



def babai_torch(bitsw, B, Btilde, v): #Babai algorithm in per-layer setting
    #bitsw is bitwidth, B is basis, Btilde is Gram-Schmidt basis, v is tensor to quantize
    n = B.shape[0]
    s = -v
    norm = torch.einsum('bs,bs->b', Btilde, Btilde)
    
    c = torch.zeros_like(v)
    a = 2**(bitsw-1)
    for i in range(n):
        B_vec = B.narrow(0,n-i-1,1)

        tmp = torch.matmul(s,Btilde.t())

        index = torch.tensor(n-i-1).cuda()
        mask = torch.zeros_like(tmp, dtype=torch.float).cuda()
        
        mask.index_fill_(-1,index,1.0)
        
        tmp = tmp * mask

        tmp = tmp / norm.narrow(0,n-i-1,1)

        tmp = q_filter.apply(tmp)
        
        s = s - tmp.sum(-1, keepdim=True)*B_vec

        tmp = torch.clamp(tmp, -a, a-1) #no bit alloc
        c = c - tmp

    return c

def loss_torch_linear(bitsw,q_w,B,do_bias_correction): #MCE per layer between weights and quantized weights
    q_w=q_w.reshape(-1,B.shape[1])
    Btilde = gram_schmidt_torch(B)
    coord = babai_torch(bitsw=bitsw, B=B, Btilde=Btilde, v=q_w)
    vect = coordinates_to_vect_torch(B=B, coordinates=coord)
    if do_bias_correction:
        vect = bias_correction(q_w, vect)

    return((torch.abs(vect.flatten()-q_w.flatten())**3).sum())
    #return F.mse_loss(vect,q_w)

def bias_correction(FP_weight,weight): #bias correction for weights (Banner et al.)
    FP_w = FP_weight.detach()
    w = weight.detach()
    ones = torch.ones_like(w)
    mu = (FP_w-w).mean(dim=(1,2),keepdim=True)*ones
    eps =  10**(-6)
    d = torch.linalg.norm((w - w.mean(dim=(1,2),keepdim=True)*ones),dim=(1,2),keepdim=True) + eps
    inv = 1/d
    eta = torch.linalg.norm((FP_w - FP_w.mean(dim=(1,2),keepdim=True)*ones),dim=(1,2),keepdim=True)*inv

    w = eta*(w+mu)
    return(w)

def loss_torch(bitsw,q_w, B, do_bias_correction): #MCE per channel between weights and quantized weights
    with torch.no_grad():
        q_w=q_w.reshape(q_w.shape[0],-1,B.shape[1])
        #print(q_w.shape)

        Btilde = gram_schmidt_torch_channel(B)

        coord = babai_torch_channel(bitsw=bitsw, B=B, Btilde=Btilde, v=q_w)
        
        vect = coordinates_to_vect_torch(B,coord)
        
        if do_bias_correction: #bias aware calibration (used for bitwidths <4)
            vect = bias_correction(q_w, vect)


    
    return((torch.abs(vect.flatten(1)-q_w.flatten(1))**3).sum((1,)))
    #return F.mse_loss(vect+shift,q_w,reduction='none').sum((1,2))

def coordinates_to_vect_torch(B, coordinates): #Convert quantized coordinates to real filters
    return coordinates.matmul(B)

def transform_ball_torch(v, sigr, cube=False): #random euclidian ball transformation
    return v+torch.normal(0,sigr,v.shape).cuda()
    #return v*torch.normal(1,sigr,(1,)).cuda() #cube

