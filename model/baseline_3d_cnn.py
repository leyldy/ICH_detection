import sys
sys.path.append('..')

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import *


class baseline_3DCNN(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(12,512,512), inter_num_ch=4, kernel_size=3, conv_act='relu',dropout=0.2):
        super().__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv3d(in_num_ch, inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm3d(inter_num_ch),
                        nn.MaxPool3d(2))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm3d(2*inter_num_ch),
                        nn.MaxPool3d(2),
                        nn.Dropout3d(dropout))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(2*inter_num_ch, 4*inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm3d(4*inter_num_ch),
                        nn.MaxPool3d(2),
                        nn.Dropout3d(dropout))

        self.conv4 = nn.Sequential(
                        nn.Conv3d(4*inter_num_ch, 2*inter_num_ch, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm3d(2*inter_num_ch),
                        nn.MaxPool3d(2))
        
        self.lin1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(65536, 128),
            nn.ReLU()
        )
        self.lin2 = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(128, 16),
                    nn.ReLU(),
        )
        self.lin3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16, 6),
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        conv_out = self.conv1(x)
#         print(conv_out.shape)
        conv_out = self.conv2(conv_out)
#         print(conv_out.shape)
        conv_out = self.conv3(conv_out)
#         print(conv_out.shape)
#         conv4 = self.conv4(conv3)
#         print(conv4.shape)
        out = flatten(conv_out)
        out = self.lin1(out)
        out = self.lin2(out)
        out = self.lin3(out)
        out = self.sigmoid(out)
        return out

