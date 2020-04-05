import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models, initializers
from tensorflow.keras import Model

import util

import skimage.measure
from torch.nn import functional as F

# from pytorch_prototyping import pytorch_prototyping
import custom_layers
# import geometry
import hyperlayers

class FCBlock(Model):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super(FCBlock, self).__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(layer .Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = Sequential()
        self.net.apply(self.init_weights)

    def __getitem__(self,item):
        return self.net[item]

    # def init_weights(self, l):
    #     if type(m) == nn.Linear:
    #         he_normal(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
    

    def forward(self, input):
        return self.net(input)
