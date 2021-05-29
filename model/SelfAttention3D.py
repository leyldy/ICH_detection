'''
This file contains the 3D self-attention layer.
This was written from scratch by Jiying Zou, with slight formatting guidance from the MultiHeadedAttention class in Assignment 3.
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
import time


class SelfAttention3D(nn.Module):
    '''
    A model layer that implements 3D self-attention as described in
    "An Explainable 3D Residual Self-Attenion Deep Neural Network for Joint Atrophy Localization
     and Altzheimer's Disease Diagnosis using Structural MRI" (https://arxiv.org/pdf/2008.04024.pdf)

    Usage:

    start = time.time()

    layer = SelfAttention3D(embed_dim = 16*5*16*16, in_channels = 16, dropout=0.1) # initialize layer
    x = torch.zeros(2, 16, 5, 16, 16) # sample input
    out = layer(x) # run thru

    end = time.time()
    print("Time consumed in working: ", end - start)

    '''

    def __init__(self, embed_dim, in_channels, dropout=0.1):
        '''
        Construct a new Attention3D layer.

        Inputs:
         - embed_dim: Dimension of the flattened input after computing attention weghts * value (C x D x H x W)
         - in_channels: Number of input channels C.
         - dropout: Dropout probability (added this to the original implementation details)
        '''

        super().__init__()
        assert embed_dim != 0
        assert in_channels != 0

        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.p_drop = dropout

        # 1x1x1 Conv3D filters to learn (filters are 1x1x1 but span all of in_channel)
        self.Wkey = nn.Conv3d(self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1)
        self.Wquery = nn.Conv3d(self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1)
        self.Wvalue = nn.Conv3d(self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1)

        # Affine linear projection at the end (matrix W_o)
        # (Note: In original paper this was another 1x1x1 3D Conv but I changed it here to linear fc for greater expressivity)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layer for right after softmax in attention calculation
        self.dropout_lyr = nn.Dropout(self.p_drop)

    def forward(self, data):
        '''
        Calculate 3D self-attention (not masked) for provided data.

        Inputs:
        - data: Input data to be used to build the query, key, and value,
                of shape (N, C, D, H, W)

        Returns:
        - output: Tensor of shape (N, C, D, H, W) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        '''

        N, C, D, H, W = data.shape

        # First convert input to keys, querys, and values
        # Put input data thru 3DConv layers with 1x1x1 filters and then flatten
        key = self.Wkey(data).view(N, -1)  # (N, E) where E = C * D * H * W
        query = self.Wquery(data).view(N, -1)  # (N, E)
        value = self.Wvalue(data).view(N, -1)  # (N, E)
        print("Key size {}".format(key.size()))

        # Get attention weights (relevance of each query to keys thru outer product of keys and queries, einsum helps ignore the N axis)
        att = F.softmax(torch.einsum('ni,nj->nij', key, query),
                        dim=1)  # (N, E, E) softmax over index i when entries are (i,j) (over columns)

        # A dropout layer right after softmax, like we did in assignment 3
        att = self.dropout_lyr(att)  # (N, E, E)
        print("Att size {}".format(att.size()))

        # Weight values with attention weights, then sum along each weighted value
        value_shape = list(value.size())[1]  # E
        output = torch.sum(att * value.reshape(N, value_shape, 1), 1)  # (N, E) sum columns
        print("Output size {}".format(output.size()))
        # ^ (N, E, E) x (N, E, 1) = (N, E, E) then sum along axis 1 = (N, E)

        # Last linear layer (learns W_o)
        output = self.proj(output)

        # Reshape back to original input dims
        output = output.view(data.size())

        return output