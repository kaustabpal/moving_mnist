# @brief:     Pytorch module for ConvLSTMcell
# @author     Kaustab Pal    [kaustab21@gmail.com]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time
#torch.manual_seed(2020701021)

class ConvLSTMCell(nn.Module):
    '''
    Peephole-Conv-LSTM cell implementation
    Peephole-LSTM paper: https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf
    Conv-LSTM paper: https://arxiv.org/pdf/1506.04214.pdf
    '''
    def __init__(self, input_dim, hidden_dim, kernel_size, padding,
            activation, frame_size, peep=True):
        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            # Using ReLU6 as ReLU as exploding grradient problems
            self.activation = nn.ReLU6() 

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.peep = peep

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim, kernel_size=kernel_size,
            padding=self.padding, bias = True
            )
        if(self.peep == True):
            # Init weights for Peep-hole connections to previous cell state
            self.W_ci = nn.Parameter(torch.randn(hidden_dim, *frame_size))
            self.W_co = nn.Parameter(torch.randn(hidden_dim, *frame_size))
            self.W_cf = nn.Parameter(torch.randn(hidden_dim, *frame_size))

    def forward(self, input_tensor, cur_state): 
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        assert not torch.any(torch.isnan(combined))

        combined_conv = self.conv(combined)
        assert not torch.any(torch.isnan(combined_conv))

        i_conv, f_conv, c_conv, o_conv =\
                torch.chunk(combined_conv, chunks=4, dim=1)

        if(self.peep == True):
            input_gate = torch.sigmoid(i_conv + self.W_ci * c_cur )
            forget_gate = torch.sigmoid(f_conv + self.W_cf * c_cur )
            # Current Cell output
            c_next = forget_gate*c_cur + input_gate * self.activation(c_conv)
            output_gate = torch.sigmoid(o_conv + self.W_co * c_next )
        else:
            input_gate = torch.sigmoid(i_conv)
            forget_gate = torch.sigmoid(f_conv)
            # Current Cell output
            c_next = forget_gate*c_cur + input_gate * self.activation(c_conv)
            output_gate = torch.sigmoid(o_conv)

        # Current Hidden State
        h_next = output_gate * self.activation(c_next)

        assert not torch.any(torch.isnan(input_gate))
        assert not torch.any(torch.isnan(forget_gate))
        assert not torch.any(torch.isnan(c_next))
        assert not torch.any(torch.isnan(output_gate))
        assert not torch.any(torch.isnan(h_next))

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, padding,
            activation, frame_size, num_layers=1,
            peep=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        padding = self._extend_for_multilayer(padding, num_layers)
        frame_size = self._extend_for_multilayer(frame_size, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.frame_size = frame_size
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        self.peep = peep

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim =\
                    self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          padding=self.padding[i],
                                          activation=activation,
                                          frame_size=self.frame_size[i],
                                          peep=self.peep))

        self.cell_list = nn.ModuleList(cell_list)
        self.norm = nn.BatchNorm2d(num_features=64)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_tensor, hidden_state=None):
        # X is a frame sequence (batch_size, seq_len, num_channels, height, width)
        # Get the dimensions
        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor = self.dropout(cur_layer_input[:, t, :, :, :]),\
                            cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
