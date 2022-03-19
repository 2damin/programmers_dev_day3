import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel, pad, stride):
        self.conv = nn.Conv2d(in_c, out_c, kernel, pad, stride, bias=False)
        self.bn = nn.Batchnorm2d(out_c)
        self.act = nn.LeakyReLU()
        self.modules = nn.Sequential([self.conv,self.bn,self.act])
    
    def forward(self, x):
        x = self.modules(x)
        return x

class ResidulBlock(nn.Module):
    def __init__(self, in_c):
        mid_c = in_c//2
        self.conv0 = ConvBlock(in_c, mid_c, 1, 0, 1)
        self.conv1 = ConvBlock(mid_c, in_c, 3, 1, 1)
    
    def forward(self, x):
        residual = x
        x = self.conv0(x)
        x = self.conv1(x)
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
        
        self.conv0 = ConvBlock(in_channel, 32, 3, 1, 1)
        self.conv1 = ConvBlock(32, 64, 3, 1, 2)
        self.resblock2 = ResidulBlock(64)
        self.conv3 = ConvBlock(64, 128, 3, 1, 2)
        self.resblock4 = self.make_layer(ResidulBlock, 128, 2)
        self.conv5 = ConvBlock(128, 256, 3, 1, 2)
        self.resblock6 = self.make_layer(ResidulBlock, 256, 8)
        self.conv7 = ConvBlock(256, 512, 3, 1, 2)
        self.resblock8 = self.make_layer(ResidulBlock, 512, 8)
        self.conv9 = ConvBlock(512, 1024, 3, 1, 2)
        self.resblock10 = self.make_layer(ResidulBlock, 1024, 4)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.Linear(1024,self.n_classes)
        
    def make_layer(self, block, in_channel, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):

        x = self.conv0(x)
        x = self.conv1(x)
        x = self.resblock2(x) 
        x = self.conv3(x)
        x = self.resblock4(x) 
        x = self.conv5(x)
        x = self.resblock6(x) 
        x = self.conv7(x)
        x = self.resblock8(x) 
        x = self.conv9(x)
        x = self.resblock10(x)
        x = self.gap(x)
        x = x.view(-1,1024)
        x = self.fc(x)
        return x
        