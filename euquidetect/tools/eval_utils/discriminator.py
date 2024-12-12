import torch
import torch.nn as nn
import torch.optim as optim

class ImageDiscriminator(nn.Module):
    def __init__(self, input_dim, pretrained_path=None):
        super(ImageDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=False),  
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.c = None  
        self.deep_svdd_loss = None
        
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        
        self.c = state_dict['c'].cuda()
        self.deep_svdd_loss = DeepSVDDLoss(self.c)  

    def forward(self, x):
        output = self.net(x)
        loss = self.deep_svdd_loss(output) if self.deep_svdd_loss else None
        return output, loss

class PointDiscriminator(nn.Module):
    def __init__(self, input_dim, pretrained_path=None):
        super(PointDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=False),  
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.c = None  
        self.deep_svdd_loss = None

        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        
        self.c = state_dict['c'].cuda()
        self.deep_svdd_loss = DeepSVDDLoss(self.c)  

    def forward(self, x):
        output = self.net(x)
        loss = self.deep_svdd_loss(output) if self.deep_svdd_loss else None
        return output, loss
    
class DeepSVDDLoss(nn.Module):
    def __init__(self, c, reduction='mean'):
        super(DeepSVDDLoss, self).__init__()
        self.c = c  
        self.reduction = reduction

    def forward(self, x):
        dist = torch.sum((x - self.c) ** 2, dim=1)
        if self.reduction == 'mean':
            return torch.mean(dist)
        elif self.reduction == 'sum':
            return torch.sum(dist)
        else:
            return dist
