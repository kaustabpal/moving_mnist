# @brief:     Generic pytorch module for NN
# @author     Kaustab Pal    [kaustab21@gmail.com]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.models.mobilenet_v3_small
import yaml
import time
from mm.models.base import BaseModel
from mm.models.conv_lstm import ConvLSTM
import random

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        B, C, H, W = l.size()
        #print(torch.cat((l,g),1).shape)
        c = self.op(l+g) # batch_sizex1xHxW
        if self.normalize_attn:
            a = F.softmax(c.view(B,1,-1), dim=2).view(B,1,H,W)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        return g, a

class Many2One(BaseModel):
    def __init__(self, cfg, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers, peep=True):
        super(Many2One, self).__init__(cfg)
        self.cfg = cfg
        self.encoder = nn.ModuleList()
        # Add First layer (Different in_channels than the rest)
        self.convLSTM = ConvLSTM(
                input_dim=num_channels, hidden_dim=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size,
                num_layers=num_layers, peep=peep, return_all_layers=True
                )
        self.norm = nn.BatchNorm3d(num_features=10)

        # Add Convolutional Layer to predict output frame
        self.conv = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

    def forward(self, X):
        # X is of shape [batch, seq_length, num_channels, height, width]
        output, h = self.convLSTM(X)
        output = output[-1]
        output = self.norm(output)
        '''
        output is a list of length num_layers
        output[i] contains the outputs of all the input seq for the i-th layer
        output[i] is of shape [batch, seq_length, num_kernels, *frame_size]
        output[-1] contains the output of the last layer

        h is a list of length num_layers
        h[i] containts the final time-step's hidden and cell state. 
        h[i][0] contains the hidden state
        h[i][1] contains the cell state
        The hidden and the cell states are of shape [b,num_kernels,*frame_size]
        '''
        #context_map = h[-1][0] # final layer's hidden state
        #assert not torch.any(torch.isnan(context_map))
        #context_map = self.norm(context_map)
        #context_map = torch.relu(context_map)
        context_map = output[:,-1] # final layer's hidden state

        # Return only the last output frame
        pred_frame = self.conv(context_map) # Using last time-step's hidden state
        assert not torch.any(torch.isnan(pred_frame))
        return pred_frame

class One2Many(BaseModel):
    def __init__(self, cfg, num_channels, num_kernels, kernel_size, padding, 
    activation, frame_size, num_layers, peep=True):
        super(One2Many, self).__init__(cfg)
        self.cfg = cfg
        self.num_kernels = num_kernels

        self.encoder = ConvLSTM(
                input_dim=num_channels, hidden_dim=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size,
                num_layers=num_layers, peep=peep, return_all_layers=True
                )
        self.conv1 = nn.ConvTranspose2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

        self.decoder = ConvLSTM(
                input_dim=num_kernels, hidden_dim=num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size,
                num_layers=num_layers, peep=peep, return_all_layers=True
                )

        self.conv2 = nn.ConvTranspose2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

        self.norm = nn.BatchNorm3d(num_features=10)

        self.attention = LinearAttentionBlock(in_features=num_kernels,
                normalize_attn=True)


    def forward(self, X, Y, teacher_forcing_ratio):
        # X is of shape [batch, seq_length, num_channels, height, width]
        batch, seq_length, num_channels, height, width = X.shape
        use_teacher_forcing = True\
                if random.random() < teacher_forcing_ratio else False
        pred_frame = torch.zeros(X.shape[0],
                self.cfg["MODEL"]["N_FUTURE_STEPS"],1,64,64).to(self.device)
        output, h = self.encoder(X)
        '''
        output is a list of length num_layers
        output[i] contains the outputs of all the input seq for the i-th layer
        output[i] is of shape [batch, seq_length, num_kernels, *frame_size]
        output[-1] contains the output of the last layer

        h is a list of length num_layers
        h[i] containts the final time-step's hidden and cell state. 
        h[i][0] contains the hidden state
        h[i][1] contains the cell state
        The hidden and the cell states are of shape [b,num_kernels,*frame_size]
        '''
        output = output[-1]
        output = self.norm(output)
        g = output[:,-1] # final layer's hidden state
        for i in range(self.cfg["MODEL"]["N_FUTURE_STEPS"]):
            context = torch.zeros((batch, self.num_kernels, height, width),
                    device=self.device) # global context vector
            for j in range(output.shape[1]):
                l = output[:,j]
                g_a, _ = self.attention(l,g)
                context += g_a
            context_map = self.conv1(context)
            context_map = context_map.unsqueeze(1)
            dec_input = context_map
            dec_output, h = self.decoder(dec_input, h)
            g = dec_output[-1][:,-1]
            dec_input = dec_output[-1][:,-1]
            dec_input = self.conv2(dec_input)
            pred_frame[:,i,:,:,:] = dec_input
            dec_input = dec_input.unsqueeze(1).detach()
            if i>0 and use_teacher_forcing:
                dec_input = Y[:,i-1].unsqueeze(1)
            # Return only the last output frame
            #pred_frame[:,i,:,:,:] = self.conv(dec_output[-1][:,-1]) # Using last time-step's hidden state
        assert not torch.any(torch.isnan(pred_frame))
        return pred_frame

if __name__ == "__main__":
    config_filename = 'config/parameters.yml'# location of config file
    cfg = yaml.safe_load(open(config_filename))
    model = One2Many(cfg, num_channels=1, num_kernels=64, 
                    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
                    frame_size=(64, 64), num_layers=3)

    #model = LinearAttentionBlock(in_features=64, normalize_attn=True)
    input = torch.randn(2,10,1,64,64)
    pred = model(input, input, 0)
    print(pred.shape)










