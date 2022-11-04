import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FDS_MPI(nn.Module):
    def __init__(self, out_ch=96):
        super(FDS_MPI, self).__init__()
        self.conv_first = nn.Conv2d(1, out_ch, kernel_size=3, stride=1, padding=0)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0)
        self.conv_t = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=1, padding=0)
        self.conv_t_last = nn.ConvTranspose2d(out_ch, 1, kernel_size=3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_ch)
        self.conv_final = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=0)
        self.downpooling = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        # self.uppooling = F.interpolate(x, scale_factor=2, mode='nearest')
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
       
        # one branch
        out = self.relu(self.bn(self.conv_first(x)))
        out = self.relu(self.bn(self.conv(out)))
        

        out = self.relu(self.bn(self.conv(out)))
        out = self.relu(self.bn(self.conv(out)))
        residual_3 = out.clone()
        out = self.relu(self.bn(self.conv(out)))
        out = self.relu(self.bn(self.conv(out)))

        out = self.relu(self.bn(self.conv(out)))
        out = self.relu(self.bn(self.conv(out)))
        residual_5 = out.clone()
        out = self.relu(self.bn(self.conv(out)))

        out = self.conv_t(self.relu(out))
        out += residual_5
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))
  
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))
        out += residual_3
        out = self.conv_t(self.relu(out))
        out = self.conv_t(self.relu(out))

        out = self.conv_t(self.relu(out))
        out = self.conv_t_last(self.relu(out))
        # two branch
        out1 = self.relu(self.conv_first(x))
        out1 = self.relu(self.conv(out1))
        out1 = self.downpooling(out1) 
        out1 = self.relu(self.conv(out1))
        out1 = self.relu(self.conv(out1))
        residual_31 = out1.clone()
        out1 = self.relu(self.conv(out1))
        out1 = self.relu(self.conv(out1))
        out1 = self.downpooling(out1)
        out1 = self.relu(self.conv(out1))
        out1 = self.relu(self.conv(out1))
        residual_51 = out1.clone()
        out1 = self.relu(self.conv(out1))

        out1 = self.conv_t(self.relu(out1))
        out1 += residual_51
        out1 = self.conv_t(self.relu(out1))
        out1 = self.conv_t(self.relu(out1))
        out1 = F.interpolate(out1, scale_factor=2, mode='nearest') 
        out1 = self.conv_t(self.relu(out1))
        out1 = self.conv_t(self.relu(out1))
        out1 += residual_31
        out1 = self.conv_t(self.relu(out1))
        out1 = self.conv_t(self.relu(out1))
        out1 = F.interpolate(out1, scale_factor=2, mode='nearest') 
        out1 = self.conv_t(self.relu(out1))
        out1 = self.conv_t_last(self.relu(out1))

        #fusion
        out_final = out+out1
        out_final = self.conv_final(out_final)
        out_final = self.relu(out_final)

        return out
