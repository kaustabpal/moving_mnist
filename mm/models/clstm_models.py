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
from mm.models.blocks import ConvLSTM, LinearAttentionBlock, CNN_encoder
from mm.models.blocks import CNN_decoder, DownBlock, UpBlock
import random


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
    activation, img_size, num_layers, peep=True):
        super(One2Many, self).__init__(cfg)
        self.cfg = cfg
        self.channels = self.cfg["MODEL"]["CHANNELS"]
        self.skip_if_channel_size = self.cfg["MODEL"]["SKIP_IF_CHANNEL_SIZE"]
        self.num_kernels = self.channels[-1]
        self.img_size = img_size
        self.cnn_encoder = CNN_encoder(self.cfg,1)
        self.cnn_decoder = CNN_decoder(self.cfg,1)
        frame_h = self.img_size[0]//2**(len(self.channels)-1)
        frame_w = self.img_size[1]//2**(len(self.channels)-1)
        self.frame_size = (frame_h, frame_w)

        self.encoder_32 = ConvLSTM(
                input_dim=32, hidden_dim=32,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=self.frame_size,
                num_layers=num_layers, peep=peep, return_all_layers=True
                )

        self.encoder = ConvLSTM(
                input_dim=self.channels[-1], hidden_dim=self.num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=self.frame_size,
                num_layers=num_layers, peep=peep, return_all_layers=True
                )
        #self.conv1 = nn.ConvTranspose2d(
        #    in_channels=num_kernels, out_channels=num_channels,
        #    kernel_size=kernel_size, padding=padding)

        self.decoder = ConvLSTM(
                input_dim=num_kernels, hidden_dim=self.num_kernels,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=self.frame_size,
                num_layers=num_layers, peep=peep, return_all_layers=True
                )
        self.decoder_32 = ConvLSTM(
                input_dim=32, hidden_dim=32,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=self.frame_size,
                num_layers=num_layers, peep=peep, return_all_layers=True
                )

        #self.conv2 = nn.ConvTranspose2d(
        #    in_channels=num_kernels, out_channels=num_channels,
        #    kernel_size=(2,2), padding=(0,0), stride= (2,2))

        self.norm = nn.BatchNorm3d(num_features=10)

        self.attention = LinearAttentionBlock(in_features=2*self.num_kernels,
                normalize_attn=True)


    def forward(self, X, Y, teacher_forcing_ratio):
        # X is of shape [batch, seq_length, num_channels, height, width]
        batch, seq_length, num_channels, height, width = X.shape
        SOS = torch.zeros(batch,num_channels, height, width, device=X.device)
        use_teacher_forcing = True\
                if random.random() < teacher_forcing_ratio else False
        pred_frame = torch.zeros(X.shape[0],
                self.cfg["MODEL"]["N_FUTURE_STEPS"],1,64,64).to(self.device)
        x, skip_32 = self.cnn_encoder(X)
        output_32, h_32 = self.encoder_32(skip_32)
        output, h = self.encoder(x)
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
        output_32 = output_32[-1]
        output_32 = self.norm(output_32)
        g_32 = output_32[:,-1]
        output = output[-1]
        output = self.norm(output)
        g = output[:,-1] # final layer's hidden state
        img = SOS
        for i in range(self.cfg["MODEL"]["N_FUTURE_STEPS"]):
            context = torch.zeros((batch, self.num_kernels,
                self.frame_size[0], self.frame_size[1]), device=self.device) # global context vector
            for j in range(output.shape[1]):
                l = output[:,j]
                l_a, _ = self.attention(l,g)
                context += l_a
            context_map = context #torch.cat((context,img),1)
            context_map = context_map.unsqueeze(1)
            #context_map = context.unsqueeze(1)
            dec_input = context_map
            dec_output, h = self.decoder(dec_input, h)
            g = dec_output[-1][:,-1]
            out_t = self.cnn_decoder(g)
            img = out_t.detach()
            # work from this part
            #img = dec_output[-1][:,-1]
            #img = self.conv2(img)
            #dec_input = dec_output[-1][:,-1]
            #dec_input = self.conv2(dec_input)
            pred_frame[:,i,:,:,:] = out_t
            #dec_input = img.unsqueeze(1).detach()
            if i>0 and use_teacher_forcing:
                img = Y[:,i-1]#.unsqueeze(1)
            # Return only the last output frame
            #pred_frame[:,i,:,:,:] = self.conv(dec_output[-1][:,-1]) # Using last time-step's hidden state
        assert not torch.any(torch.isnan(pred_frame))
        return pred_frame

class SkipModel(BaseModel):
    def __init__(self, cfg, num_channels, num_kernels, kernel_size, padding, 
    activation, img_size, num_layers, peep=True):
        super(SkipModel, self).__init__(cfg)
        self.cfg = cfg
        self.channels = self.cfg["MODEL"]["CHANNELS"]
        self.skip_if_channel_size = self.cfg["MODEL"]["SKIP_IF_CHANNEL_SIZE"]
        self.num_kernels = self.channels[-1]
        self.img_size = img_size
        #self.cnn_encoder = CNN_encoder(self.cfg,1)
        #self.cnn_decoder = CNN_decoder(self.cfg,1)
        frame_h = self.img_size[0]//2**(len(self.channels)-1)
        frame_w = self.img_size[1]//2**(len(self.channels)-1)
        self.frame_size = (frame_h, frame_w)

        self.input_layer = nn.Conv2d(in_channels=num_channels,
                out_channels=self.channels[0],
                kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)

        self.DownLayers = nn.ModuleList()
        self.UpLayers = nn.ModuleList()

        for i in range(len(self.channels) - 1):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.DownLayers.append(
                        DownBlock(
                            self.channels[i],
                            self.channels[i + 1],
                            skip=True,
                        )
                    )
            else:
                self.DownLayers.append(
                        DownBlock(
                            self.channels[i],
                            self.channels[i + 1],
                            skip=False,
                        )
                    )

        for i in reversed(range(len(self.channels) - 1)):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.UpLayers.append(
                    UpBlock(
                        self.channels[i + 1],
                        self.channels[i],
                        skip=True,
                    )
                )
            else:
                self.UpLayers.append(
                    UpBlock(
                        self.channels[i + 1],
                        self.channels[i],
                        skip=False,
                    )
                )
        self.output_layer = nn.Conv2d(in_channels=self.channels[0],
                out_channels=1,
                kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention = nn.ModuleList()
        for i in range(len(self.channels)):
            if self.channels[i] in self.skip_if_channel_size:
                self.encoder.append(
                        ConvLSTM(
                            input_dim=self.channels[i],
                            hidden_dim=self.channels[i],
                            kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            frame_size=self.frame_size, num_layers=num_layers,
                            peep=peep, return_all_layers=True)
                        )
                self.attention.append(
                        LinearAttentionBlock(in_features=self.channels[i],
                            normalize_attn=True)
                        )
                self.decoder.append(
                        ConvLSTM(
                            input_dim=self.channels[i],
                            hidden_dim=self.channels[i],
                            kernel_size=kernel_size,
                            padding=padding, activation=activation,
                            frame_size=self.frame_size, num_layers=num_layers,
                            peep=peep, return_all_layers=True)
                        )
        self.encoder.append(
                ConvLSTM(
                    input_dim=self.channels[-1],
                    hidden_dim=self.channels[-1],
                    kernel_size=kernel_size,
                    padding=padding, activation=activation,
                    frame_size=self.frame_size, num_layers=num_layers,
                    peep=peep, return_all_layers=True)
                )
        self.attention.append(
                LinearAttentionBlock(in_features=self.channels[-1],
                    normalize_attn=True)
                )
        self.decoder.append(
                ConvLSTM(
                    input_dim=self.channels[-1],
                    hidden_dim=self.channels[-1],
                    kernel_size=kernel_size,
                    padding=padding, activation=activation,
                    frame_size=self.frame_size, num_layers=num_layers,
                    peep=peep, return_all_layers=True)
                )

        self.norm = nn.BatchNorm3d(num_features=10)

    def forward(self, X, Y, teacher_forcing_ratio):
        # X is of shape [batch, seq_length, num_channels, height, width]
        device = X.device
        batch, seq_length, num_channels, height, width = X.shape
        SOS = torch.zeros((batch,num_channels, height, width), device=device)
        use_teacher_forcing = True\
                if random.random() < teacher_forcing_ratio else False
        pred_frame = torch.zeros(X.shape[0],
                self.cfg["MODEL"]["N_FUTURE_STEPS"],1,64,64).to(device)
        skip_layer_encoder = []
        skip_layer_decoder = []
        attn_list = []
        encoder_output = []
        encoder_hidden = []
        X = X.view(batch*seq_length, num_channels, height, width).to(device)
        X = self.input_layer(X)
        for l in self.DownLayers:
            X = l(X)
            if l.skip:
                skip_layer_encoder.append(X)
        skip_layer_encoder.append(X)
        for s in range(len(skip_layer_encoder)):
            _,c,h,w = skip_layer_encoder[s].shape
            skip_layer_encoder[s] =\
                    skip_layer_encoder[s].view(batch,seq_length,c,h,w).to(device)
            output, h = self.encoder[s](skip_layer_encoder[s])
            output = output[-1]
            output = self.norm(output)
            g = output[:,-1] # final layer's hidden state
            g_out = torch.zeros((batch, self.cfg["MODEL"]["N_FUTURE_STEPS"],
                    g.shape[1], g.shape[2], g.shape[3]),device=device)
            for i in range(self.cfg["MODEL"]["N_FUTURE_STEPS"]):
                context = torch.zeros((batch, g.shape[1],
                    g.shape[2], g.shape[3]),
                    device=device) # global context vector
                a_out, attn = self.attention[s](output,g)
                attn_list.append(attn)
                for t in range(a_out.shape[1]):
                    context +=a_out[:,t]
                #for j in range(output.shape[1]):
                #    l = output[:,j]
                #    l_a, _ = self.attention[s](l,g)
                #    context += l_a
                context_map = context #torch.cat((context,img),1)
                dec_input = context_map.unsqueeze(1)
                dec_output, h = self.decoder[s](dec_input, h)
                g = dec_output[-1][:,-1]
                g_out[:,i] = g
            skip_layer_decoder.append(
                    g_out.view(
                        batch*self.cfg["MODEL"]["N_FUTURE_STEPS"],
                        g_out.shape[2],g_out.shape[3],g_out.shape[4]).to(device)
                    )

        x = skip_layer_decoder.pop()
        for l in self.UpLayers:
            if l.skip:
                x = l(x, skip_layer_decoder.pop())
            else:
                x = l(x)
        x = self.output_layer(x)
        pred_frame = x.view(batch,self.cfg["MODEL"]["N_FUTURE_STEPS"], x.shape[1], x.shape[2], x.shape[3]).to(device)
        assert not torch.any(torch.isnan(pred_frame))
        return pred_frame, attn_list

if __name__ == "__main__":
    config_filename = 'config/parameters.yml'# location of config file
    cfg = yaml.safe_load(open(config_filename))
    model = SkipModel(cfg, num_channels=1, num_kernels=32, 
                    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
                    img_size=(64, 64), num_layers=3, peep=False)
    model = model.to("cuda")

    #model = LinearAttentionBlock(in_features=64, normalize_attn=True)
    input = torch.randn(2,10,1,64,64).to("cuda")
    pred = model(input, input, 0)
    print(pred.shape)










