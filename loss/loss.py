import torch
from torch import nn
import sys

class BCEloss(nn.Module):
    def __init__(self, device = torch.device('cpu'), weight = None):
        super(BCEloss, self).__init__()
        self.loss =nn.BCELoss().to(device)

    def forward(self, out, gt):
        loss_val = self.loss(out, gt)
        return loss_val
    
def get_criterion(crit = "bce", device = torch.device('cpu'),  weight = None):
    if crit == "bce":
        return BCEloss(device = device, weight = weight)
    else:
        print("unknown criterion")
        sys.exit(1)
