import torch
from torch import nn
import sys

class BCEloss(nn.Module):
    def __init__(self, device = torch.device('cpu')):
        super(BCEloss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device)
    
    def forward(self, out, gt):
        loss_val = self.loss(out, gt)
        return loss_val
    
def get_criterion(crit = "bce", device = torch.device('cpu')):
    if crit == "bce":
        return BCEloss(device = device)
    else:
        print("unknown criterion")
        sys.exit(1)

