from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

class FocalLoss(nn.Module):
    
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, target, output):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            f'Length of weight tensor must match the number of classes. got {num_classes} expected {len(self.alpha)}'
        logp = F.cross_entropy(output, target, self.alpha)
        p = torch.exp(-logp)
        focal_loss = (1-p)**self.gamma*logp
 
        return torch.mean(focal_loss)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, device, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        print("LDAM weights:", weight)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list.cpu()))
        #m_list = m_list * (max_m / np.max(m_list))
        m_list = (m_list * (max_m / torch.max(m_list))).to(device)
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.device = device
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self,target, x):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
        #criterion = LDAMLoss(cls_num_list=a list of numer of samples in each class, max_m=0.5, s=30, weight=per_cls_weights)


class LMFLoss(nn.Module):
        def __init__(self, cls_num_list, device, weight, alpha=0.2, beta=0.2, gamma=2, max_m=1.0, s=10, add_LDAM_weigth=True): 
            super().__init__()
            self.focal_loss = FocalLoss(weight, gamma)
            if add_LDAM_weigth:
                LDAM_weight = weight
            else:
                LDAM_weight = None
            print("LMF loss: alpha: ", alpha, " beta: ", beta, " gamma: ", gamma, " max_m: ", max_m, " s: ", s, " LDAM_weight: ", add_LDAM_weigth)
            self.ldam_loss = LDAMLoss(cls_num_list, device, max_m, weight=LDAM_weight, s=s)
            self.alpha= alpha
            self.beta = beta

        def forward(self, target, output):
            focal_loss_output = self.focal_loss(target, output)
            ldam_loss_output = self.ldam_loss(target, output)
            total_loss = self.alpha*focal_loss_output + self.beta*ldam_loss_output
            return total_loss 