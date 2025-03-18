import os
import sys
import time
import math

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
        

def kd( y_s, y_t, is_ca=False):
        p_s = F.log_softmax(y_s/4, dim=1)
        p_t = F.softmax(y_t/4, dim=1)
        if is_ca: 
            loss = (nn.KLDivLoss(reduction='none')(p_s, p_t) * (4**2)).sum(-1)
        else:
            loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (4**2)
        return loss



def MASD(net, inputs, targets, criterion_cls, criterion_div,epoch,args):
    loss_div = torch.tensor(0.).cuda()
    loss_cls = torch.tensor(0.).cuda()
    
    #alpha_t = 1 - epoch / (args.epochs - 1)
    #alpha_t = max(0, alpha_t) 
    
    outputs = net(inputs,use_auxiliary=True)


    criterion_cls_lc = nn.CrossEntropyLoss(reduction='none')
    #loss_cls += criterion_cls(outputs[0], targets)
    #loss_cls += alpha_t * criterion_cls(outputs[1], targets)
    #loss_cls += alpha_t * criterion_cls(outputs[2], targets)

    for i, output_s in enumerate(outputs):
        loss_cls += criterion_cls(output_s, targets)
        
        teachers = [logit_t for j, logit_t in enumerate(outputs) if j != i]
        
        loss_t_list = [criterion_cls_lc(logit_t, targets) for logit_t in teachers]

        loss_t = torch.stack(loss_t_list, dim=0)
  
        attention = (1.0 - F.softmax(loss_t, dim=0)) / (2 - 1)
        
        loss_div_list = [kd(output_s, logit_t,is_ca=True) for logit_t in teachers]
        loss_div1 = torch.stack(loss_div_list, dim=0)
   
    
        bsz = loss_div1.shape[1]
    
    
        loss_div += 1.5*(torch.mul(attention, loss_div1).sum()) / (1.0 * bsz * 2)
    
    
    logit = outputs[0]

    return logit, loss_cls, loss_div

