import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, in_num_ch=1, img_size=(32,64,64), inter_num_ch=16, kernel_size=3, conv_act='relu',dropout=0.2):
        super(FeatureExtractor, self).__init__()


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


    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        return conv4


classifier = nn.Sequential(
                    nn.Dropout(args.dropout),
                    nn.Linear(2048, 128),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(128, 16),
                    nn.ReLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(16, 2),
                ).to(device)