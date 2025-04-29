import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as td
import numpy as np
import matplotlib.pyplot as plt
from get_loss import LMFLoss 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ce_loss(p, alpha, c):
    # p = p.unsqueeze(dim=1)
    label = F.one_hot(p, num_classes=c)
    return  F.cross_entropy(alpha, label.float())

def weight_cls(cls_num_list):
    beta = 0.9999
    cls_num_list = [cls_num_list[i] for i in range(len(cls_num_list))]
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
    cls_num_list = torch.FloatTensor(cls_num_list).to(device)

    return per_cls_weights, cls_num_list

class RIML(nn.Module):
    def __init__(self, classes, category_counts, views, classifier_dims, annealing_epochs=1):
        super(RIML, self).__init__()
        self.views = views
        self.classes = classes
        self.annealing_epochs = annealing_epochs
        self.category_counts = category_counts
        self.Net = nn.ModuleList([Net(classifier_dims[i], self.classes) for i in range(self.views)])
        self.per_cls_weights, self.category_counts = weight_cls(category_counts)
        self.loss = LMFLoss(cls_num_list= self.category_counts, device=device, weight = self.per_cls_weights)

    def classify(self, input , y, global_step, sn):
        # pretrain
        evidence = self.collect(input)
        loss_class = 0
        alpha = dict()
        for v_num in range(self.views):
            sn_v = torch.unsqueeze(sn[:, v_num], 1)
            alpha[v_num] = evidence[v_num]
            # loss_class += ce_loss(y, alpha[v_num], self.classes) * sn_v
            loss_class += self.loss(y, alpha[v_num]) * sn_v
        loss_class = torch.mean(loss_class)
        return loss_class

    def forward(self, input, y, global_step, batch_idx, sn, if_test=0):
        if(if_test):
            evidence = self.collect(input)
            alpha = dict()
            for v_num in range(self.views):
                alpha[v_num] = evidence[v_num]
            evidence_a = torch.mean(torch.stack(list(alpha.values())), dim=0)
            return evidence, evidence_a
        evidence = self.collect(input)
        loss = 0
        alpha = dict()
        for v_num in range(self.views):
            # sn_v = torch.unsqueeze(sn[:, v_num], 1)
            alpha[v_num] = evidence[v_num]
            loss += self.loss(y, alpha[v_num]) # 分视角特征
        alpha_a = torch.mean(torch.stack(list(alpha.values())), dim=0)

        loss+= self.loss(y, alpha_a)
        loss = torch.mean(loss)
        return loss
    
    def val(self, input, y, global_step, batch_idx):
        evidence = self.collect(input)
        loss = 0
        alpha = dict()
        for v_num in range(self.views):
            # sn_v = torch.unsqueeze(sn[:, v_num], 1)
            alpha[v_num] = evidence[v_num]
            loss += self.loss(y, alpha[v_num])
        alpha_a = torch.mean(torch.stack(list(alpha.values())), dim=0)

        loss+= self.loss(y, alpha_a)
        loss = torch.mean(loss)
        return loss


    def collect(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Net[v_num](input[v_num])
        return evidence

class Net(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Net, self).__init__()
        self.classes = classes
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(classifier_dims[0], 64))
        self.fc.append(nn.Sigmoid())
        self.fc.append(nn.Dropout(0.2))

        self.evidence = nn.ModuleList()
        self.evidence.append(nn.Linear(64, classes))
        self.evidence.append(nn.Softplus())

    def forward(self, x):

        out = self.fc[0](x)
        for i in range(1, len(self.fc)):
            out = self.fc[i](out)

        evidence = self.evidence[0](out)
        evidence = self.evidence[1](evidence)

        return evidence*evidence

