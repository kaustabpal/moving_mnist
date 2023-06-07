# @brief:     Generic pytorch module for NN
# @author     Kaustab Pal    [kaustab21@gmail.com]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time
torch.manual_seed(2020701021)
from mm.models.base import BaseModel
from mm.models.conv_lstm import ConvLSTM

class Many2One(BaseModel):
    def __init__(self, cfg, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers):
        super(Many2One, self).__init__(cfg)
        self.cfg = cfg
        self.encoder = nn.ModuleList()
        # Add First layer (Different in_channels than the rest)
        self.convLSTM = ConvLSTM(
                input_dim=num_channels, hidden_dim=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size,
                num_layers=num_layers, return_all_layers=True
                )
        #self.norm = nn.BatchNorm3d(num_features=10)

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        # X is of shape [batch, seq_length, num_channels, height, width]
        output, h = self.convLSTM(X)
        '''
        output is a list of length num_layers
        output[i] contains the outputs of all the input seq for the i-th layer
        output[i] is of shape [batch, seq_length, num_kernels, *frame_size]
        output[-1] contains the output of the last layer

        h is a list of length num_layers
        h[i] containts the final time-step's hidden and cell state. 
        h[i][0] contains the hidden state
        h[i][1] contains the cell state
        The hidden and the cell states are of shape [b,num_kernels, *frame_size]
        '''
        context_map = h[-1][0] # final layer's hidden state
        assert not torch.any(torch.isnan(context_map))

        # Return only the last output frame
        pred_frame = self.conv(context_map) # Using last time-step's hidden state
        assert not torch.any(torch.isnan(pred_frame))
        #return nn.Sigmoid()(output)
        return nn.Sigmoid()(pred_frame)

#class One2Many(BaseModel):
#    def __init__(self, cfg, num_channels, num_kernels, kernel_size, padding, 
#    activation, frame_size, num_layers):
#        super(Many2One, self).__init__(cfg)
#        self.cfg = cfg
#        self.sequential = nn.Sequential()
#        # Add First layer (Different in_channels than the rest)
#        self.sequential.add_module(
#            "convlstm1", ConvLSTM(
#                in_channels=num_channels, out_channels=num_kernels,
#                kernel_size=kernel_size, padding=padding, 
#                activation=activation, frame_size=frame_size)
#        )
#        self.sequential.add_module(
#            "batchnorm1", nn.BatchNorm3d(num_features=10)
#        ) 
#        # Add rest of the layers
#        for l in range(2, num_layers+1):
#            self.sequential.add_module(
#                f"convlstm{l}", ConvLSTM(
#                    in_channels=num_kernels, out_channels=num_kernels,
#                    kernel_size=kernel_size, padding=padding, 
#                    activation=activation, frame_size=frame_size)
#                )
#            self.sequential.add_module(
#                f"batchnorm{l}", nn.BatchNorm3d(num_features=10)
#                ) 
#        # Add Convolutional Layer to predict output frame
#        self.conv = nn.Conv2d(
#            in_channels=num_kernels, out_channels=num_channels,
#            kernel_size=kernel_size, padding=padding)
#
#    def forward(self, X):
#        # Forward propagation through all the layers
#        output, h = self.sequential(X)
#        # Return only the last output frame
#        output = self.conv(output[:,-1, :]) # Using last time-step's hidden state
#        #return nn.Sigmoid()(output)
#        return output, h
#
#class Many2Many(BaseModel):
#    def __init__(self, cfg, num_channels, num_kernels, kernel_size, padding, 
#    activation, frame_size, num_layers):
#        super(Many2Many, self).__init__(cfg)
#        self.cfg = cfg
#        self.encoder = nn.Sequential()
#        # Add First layer (Different in_channels than the rest)
#        self.encoder.add_module(
#            "convlstm1", ConvLSTM(
#                in_channels=num_channels, out_channels=num_kernels,
#                kernel_size=kernel_size, padding=padding, 
#                activation=activation, frame_size=frame_size)
#        )
#        self.encoder.add_module(
#            "batchnorm1", nn.BatchNorm3d(num_features=10)
#        ) 
#        # Add rest of the layers
#        for l in range(2, num_layers+1):
#            self.encoder.add_module(
#                f"convlstm{l}", ConvLSTM(
#                    in_channels=num_kernels, out_channels=num_kernels,
#                    kernel_size=kernel_size, padding=padding, 
#                    activation=activation, frame_size=frame_size)
#                )
#            self.sequential.add_module(
#                f"batchnorm{l}", nn.BatchNorm3d(num_features=10)
#                ) 
#        # Add Convolutional Layer to predict output frame
#        self.conv = nn.Conv2d(
#            in_channels=num_kernels, out_channels=num_channels,
#            kernel_size=kernel_size, padding=padding)
#
#    def forward(self, X):
#        # Forward propagation through all the layers
#        context = self.encoder(X)[:,-1, :]
#        # Return only the last output frame
#        output = self.conv(context) # Using last time-step's hidden state
#        #return nn.Sigmoid()(output)
#        return output

if __name__ == "__main__":
    config_filename = 'config/parameters.yml'# location of config file
    cfg = yaml.safe_load(open(config_filename))
    model = Many2One(cfg, num_channels=1, num_kernels=64, 
                    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
                    frame_size=(64, 64), num_layers=3)
    input = torch.randn(2,10,1,64,64)
    pred = model(input)
    print(pred.shape)
