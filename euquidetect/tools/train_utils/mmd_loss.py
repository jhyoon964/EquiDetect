import torch
import torch.nn as nn
import torch.nn.functional as F

class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def forward(self, source, target):
        source_mean = torch.mean(source, dim=0)
        target_mean = torch.mean(target, dim=0)
        mmd = F.mse_loss(source_mean, target_mean)
        return mmd
