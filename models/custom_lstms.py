'''
The following code is heavily modified from the original found here:
https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py
'''

import numbers
import warnings
from collections import namedtuple
from typing import List, Tuple, Final

import torch
# import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from models.probact import EWTrainableMuSigma 


"""
Some helper classes for writing custom TorchScript LSTMs.

Goals:
- Classes are easy to read, use, and extend
- Performance of custom LSTMs approach fused-kernel-levels of speed.

A few notes about features we could add to clean up the below code:
- Support enumerate with nn.ModuleList:
  https://github.com/pytorch/pytorch/issues/14471
- Support enumerate/zip with lists:
  https://github.com/pytorch/pytorch/issues/15952
- Support overriding of class methods:
  https://github.com/pytorch/pytorch/issues/10733
- Support passing around user-defined namedtuple types for readability
- Support slicing w/ range. It enables reversing lists easily.
  https://github.com/pytorch/pytorch/issues/10774
- Multiline type annotations. List[List[Tuple[Tensor,Tensor]]] is verbose
  https://github.com/pytorch/pytorch/pull/14922
"""

# Probabilistic custom LSTM

def prob_lstm(
    num_features,
    hidden_size,
    num_layers,
    batch_first=False,
    dropout=0.1,
    prob_params={}
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    stack_type = MyStackedLSTMWithDropout
    layer_type = LSTMLayer
    return stack_type(
        num_layers,
        layer_type,
        batch_first,
        dropout,
        first_layer_args=[ProbLSTMCell, num_features, hidden_size, prob_params],
        other_layer_args=[ProbLSTMCell, hidden_size, hidden_size, prob_params],
    )


 # Non-probabilistic custom LSTM

def my_lstm(
    num_features,
    hidden_size,
    num_layers,
    batch_first=False,
    dropout=0.1
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    stack_type = MyStackedLSTMWithDropout
    layer_type = LSTMLayer
    return stack_type(
        num_layers,
        layer_type,
        batch_first,
        dropout,
        first_layer_args=[LSTMCell, num_features, hidden_size],
        other_layer_args=[LSTMCell, hidden_size, hidden_size],
    )


LSTMState = namedtuple("LSTMState", ["hx", "cx"])


class ProbLSTMCell(nn.Module):
    def __init__(self, num_features, hidden_size, prob_params):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, num_features))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.prob_params = {
        'mean_mu':0.0,
        'std_mu':0.0,
        'mean_sigma':0.0,
        'std_sigma':0.0,
        # to use a sigmoid for sigma, set the two following to diffrent than 0.0
        # Both are set as non-trainable in the probact.py file
        'alpha':0.0, # must not be integer
        'beta':0.0 # must not be integer
        }
        
        for k, v in prob_params.items():
            if k in self.prob_params:
                self.prob_params[k] = v
        
        self.sigma = EWTrainableMuSigma([4 * hidden_size],self.prob_params)

        #self.sigma_ih = EWTrainableMuSigma([4 * hidden_size, num_features],**init_params)
        #self.sigma_hh = EWTrainableMuSigma([4 * hidden_size, hidden_size],**init_params)

        #self.sigma_ih = Parameter(torch.zeros_like(self.weight_ih))
        #self.sigma_hh = Parameter(torch.zeros_like(self.weight_hh))

    def forward(
        self, inp: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        hx, cx = state
        #eps = torch.normal(mean = 0.0, std = 1.0, size = (hx.size(0),4*hx.size(1))).to(self.device)
        #print(eps.size())
        gates = (
            torch.mm(inp, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        gates_NAME = gates + self.sigma(gates)
        #print(self.sigma(gates).size()) # gives [4*hidden_size, batch_size]
        #print(self.sigma(gates).mean())
        ingate, forgetgate, cellgate, outgate = gates_NAME.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        
        return hy, (hy, cy)


class LSTMCell(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, num_features))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(
        self, inp: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (
            torch.mm(inp, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    def forward(
        self, inp: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = inp.unbind(0)
        outputs: List[Tensor] = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)


class MyStackedLSTMWithDropout(nn.Module):
    # Necessary for iterating through self.layers and dropout support
    # __constants__ = ["layers", "num_layers"] # change to Final, for use with torch.compile:
    layers: Final[nn.ModuleList]
    num_layers: Final[int]

    def __init__(self, num_layers, layer, batch_first, dropout,
                 first_layer_args, other_layer_args):
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )
        
        # Gab: This is to create the states in the forward
        self.hidden_size = first_layer_args[2]

        # Introduces a Dropout layer on the outputs of each LSTM layer except
        # the last layer, with dropout probability = dropout.
        self.num_layers = num_layers

        if num_layers == 1:
            warnings.warn(
                "dropout lstm adds dropout layers after all but last "
                "recurrent layer, it expects num_layers greater than "
                "1, but got num_layers = 1"
            )

        self.batch_first = batch_first
        print('LSTM dropout',dropout)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, inp: Tensor # Gab removed this:  inp: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        
        # Gab: First, I will check batch_first, and change inp to [seq_len, batch, num_features] if necessary
        if self.batch_first:
            inp = torch.swapaxes(inp,0,1)
        shape_inp = inp.size()
        batch = shape_inp[1]

        # Gab: Now, I create the states 
        device = inp.device
        states = create_states(batch, self.hidden_size, self.num_layers)
        states = [LSTMState(i.hx.to(device), i.cx.to(device)) for i in states]
        #states = [i.to(device) for i in states]
        # print('states in device',states[0][0].get_device())

        # List[LSTMState]: One state per layer
        output_states: List[Tuple[Tensor, Tensor]] = []
        output = inp
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1
        return output, output_states


def create_states(batch: int, hidden_size: int, num_layers: int):
    # Cannot implement this line!
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    states = [ # Gab: Was torch.rand, but the standard is torch.zeros
        LSTMState(torch.zeros(batch, hidden_size), torch.zeros(batch, hidden_size))
        for _ in range(num_layers)
    ]
    return states