
import torch
import math

class EndForward(Exception):
    pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def optimizer(model,opt):
    model_params=[]
    for n,p in model.named_parameters():
        if 'weight' in n:
            lr=opt.lr
            item = {'params':p,'lr':lr,'amsgrad':True}
            model_params.append(item)
        elif 'base' in n:
            lr=opt.lr
            item = {'params':p,'lr':lr,'amsgrad':True}
            model_params.append(item)
        elif 'alpha' in n: 
            lr=opt.alpha_lr
            item = {'params':p,'lr':lr,'momentum':0.9,'nesterov':True}
            model_params.append(item)
    return(torch.optim.Adam(model_params))

def Hook_activation2(opt, model, name, parent_name=''):
    for module_name in model._modules:
        m = model._modules[module_name]
        if m is not None :
            if parent_name+'_'+module_name == name:
                opt.handles.append(m.register_forward_hook(get_activation(dict=opt.activation, name=name)))
            elif len(m._modules) > 0:
                Hook_activation2(opt, m, name, parent_name+'_'+module_name)

def forward_activation(opt, model, x, name, init_hook):

    #Place the hook the collect activations of FP model
    if init_hook:
        Hook_activation2(opt, model, name)

    try:
        model(x)
    
    except EndForward:
        pass

def get_activation(dict, name):
    def hook(model, input, output):
        dict[name+'_output'] = output
        raise(EndForward) #end forward pass
    return hook

def adjust_learning_rate(opt, optimizer, step, nb_step): #learning rate cosine annealing

    eta_min = opt.lr * (.1)
    new_lr = eta_min + (opt.lr - eta_min) * (1 + math.cos(math.pi * step / nb_step)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    return new_lr

