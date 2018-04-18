from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class SFD(nn.Module):
    def __init__(self, num_layers, num_channels, num_input_channels):
        self.num_layers = num_layers

        self.convs = []
        for i in range(self.num_layers):
            in_channels = num_input_channels if i == 0 else num_channels
            out_channels = num_input_channels if i == self.num_layers - 1 else num_channels
            self.convs.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))

        self.relu = nn.ReLU(True)

        self.module_list = nn.ModuleList(self.convs)

    def forward(self, x):
        features = []
        out = x
        for i in range(self.num_layers):
            out = self.convs[i](out)
            if i != self.num_layers - 1:
                out = self.relu(out)
            features.append(out)
        return features



class MFD(nn.Module):
    def __init__(self, num_layers, num_channels, num_input_channels):
        self.num_layers = num_layers

        self.convs = []
        self.channels = []
        for i in range(self.num_layers):
            if i == 0:
                in_channels = 2 * num_channels
            elif i != self.num_layers - 1:
                in_channels = 3 * num_channels
            else:
                in_channels = num_channels + 2 * num_input_channels
    
            out_channels = num_input_channels if i == self.num_layers - 1 else num_channels
            
            self.channels.append(out_channels)
            self.convs.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))

        self.relu = nn.ReLU(True)

        self.module_list = nn.ModuleList(self.convs)

        self.previous_features = None

    def clear_features(self):
        for f in self.previous_features:
            f.zero_()

    def init_features(self, batch_size, height, width):
        self.previous_features = []
        for i in range(self.num_layers):
            f = Variable(torch.FloatTensor(batch_size, 
                                           self.channels[i], 
                                           height, 
                                           width).zero_())
            self.previous_feature.append(f)


    def forward(self, x, sf_features):
        out = x
        for i in range(self.num_layers):
            cat_input = [sf_features[i], self.previous_features[i]]
            if i != 0:
                cat_input += [out]
            out = self.convs[i](out)
            if i != self.num_layers - 1:
                out = self.relu(out)
            self.previous_features[i] = out
        return features