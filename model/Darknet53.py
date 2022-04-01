import torch
from torch import nn

def ConvBlock(in_c, out_c, kernel, pad, stride):
        conv = nn.Conv2d(in_c, out_c, kernel, stride, pad, bias=False)
        bn = nn.BatchNorm2d(out_c)
        act = nn.LeakyReLU()
        return nn.Sequential(*[conv,bn,act])

class ResidulBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        mid_c = in_c//2
        self.layer1 = ConvBlock(in_c, mid_c, 1, 0, 1)
        self.layer2 = ConvBlock(mid_c, in_c, 3, 1, 1)
    
    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        x += residual
        return x
    
class DarkNet53(nn.Module):
    def __init__(self, batch, n_classes, in_channel, in_width, in_height, is_train = False):
        super().__init__()
        self.batch = batch
        self.n_classes = n_classes
        self.in_width = in_width
        self.in_height = in_height
        self.in_channel = in_channel
        self.is_train = is_train
        
        self.conv1 = ConvBlock(in_channel, 32, 3, 1, 1)
        self.conv2 = ConvBlock(32, 64, 3, 1, 2)
        self.residual_block1 = self.make_layer(ResidulBlock, 64, 1)
        self.conv3 = ConvBlock(64, 128, 3, 1, 2)
        self.residual_block2 = self.make_layer(ResidulBlock, 128, 2)
        self.conv4 = ConvBlock(128, 256, 3, 1, 2)
        self.residual_block3 = self.make_layer(ResidulBlock, 256, 8)
        self.conv5 = ConvBlock(256, 512, 3, 1, 2)
        self.residual_block4 = self.make_layer(ResidulBlock, 512, 8)
        self.conv6 = ConvBlock(512, 1024, 3, 1, 2)
        self.residual_block5 = self.make_layer(ResidulBlock, 1024, 4)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024,self.n_classes)
        
    def make_layer(self, block, in_channel, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_block1(x)
        x = self.conv3(x)
        x = self.residual_block2(x) 
        x = self.conv4(x)
        x = self.residual_block3(x) 
        x = self.conv5(x)
        x = self.residual_block4(x) 
        x = self.conv6(x)
        x = self.residual_block5(x)
        x = self.gap(x)
        x = x.view(-1,1024)
        x = self.fc(x)
        return x
        