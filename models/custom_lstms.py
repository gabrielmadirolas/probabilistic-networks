'''
This original file (stored in same folder, with the name ORIGINAL_custom_lstms.py)
was downloaded from here:
https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py
The code below may contain modifications made by Gabriel Madirolas
'''

import numbers
import warnings
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.jit as jit
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from .probact import EWTrainableMuSigma 


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


def prob_lstm(
    num_features,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=0.1,
    bidirectional=False
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    assert bias
    # stack_type = StackedLSTMWithDropout # Gab: changed this for the following
    stack_type = MyStackedLSTMWithDropout
    layer_type = LSTMLayer
    dirs = 1 # Gab: Because in the older versions (see below) there's the option for bidirectional LSTM
    return stack_type(
        num_layers,
        layer_type,
        batch_first, # Gab: added this
        dropout, # Gab: added this
        first_layer_args=[ProbLSTMCell, num_features, hidden_size],
        other_layer_args=[ProbLSTMCell, hidden_size * dirs, hidden_size],
    )


 # Gab: my version of script_lstm

def my_lstm(
    num_features,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=0.1,
    bidirectional=False
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    assert bias
    # stack_type = StackedLSTMWithDropout # Gab: changed this for the following
    stack_type = MyStackedLSTMWithDropout
    layer_type = LSTMLayer
    dirs = 1 # Gab: Because in the older versions (see below) there's the option for bidirectional LSTM
    return stack_type(
        num_layers,
        layer_type,
        batch_first, # Gab: added this
        dropout, # Gab: added this
        first_layer_args=[LSTMCell, num_features, hidden_size],
        other_layer_args=[LSTMCell, hidden_size * dirs, hidden_size],
    )


# Old version of the above function
'''
def my_lstm(
    num_features,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=False,
    bidirectional=False,
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    assert bias
    # assert not batch_first # Gab: remove this line when batch_first is implemented

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    elif dropout:
        # stack_type = StackedLSTMWithDropout # Gab: changed this for the following
        stack_type = MyStackedLSTMWithDropout
        layer_type = LSTMLayer
        dirs = 1
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(
        num_layers,
        layer_type,
        batch_first, # Gab: added this
        first_layer_args=[LSTMCell, num_features, hidden_size],
        other_layer_args=[LSTMCell, hidden_size * dirs, hidden_size],
    )
'''
    

def script_lstm(
    num_features,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=False,
    bidirectional=False,
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    assert bias
    assert not batch_first

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    elif dropout:
        stack_type = StackedLSTMWithDropout
        layer_type = LSTMLayer
        dirs = 1
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[LSTMCell, num_features, hidden_size],
        other_layer_args=[LSTMCell, hidden_size * dirs, hidden_size],
    )


def script_lnlstm(
    num_features,
    hidden_size,
    num_layers,
    bias=True,
    batch_first=False,
    dropout=False,
    bidirectional=False,
    decompose_layernorm=False,
):
    """Returns a ScriptModule that mimics a PyTorch native LSTM."""

    # The following are not implemented.
    assert bias
    assert not batch_first
    assert not dropout

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(
        num_layers,
        layer_type,
        first_layer_args=[
            LayerNormLSTMCell,
            num_features,
            hidden_size,
            decompose_layernorm,
        ],
        other_layer_args=[
            LayerNormLSTMCell,
            hidden_size * dirs,
            hidden_size,
            decompose_layernorm,
        ],
    )


LSTMState = namedtuple("LSTMState", ["hx", "cx"])


def reverse(lst: List[Tensor]) -> List[Tensor]:
    return lst[::-1]


class ProbLSTMCell(jit.ScriptModule):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, num_features))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        init_params = {
        'mean_mu':0.0,
        'std_mu':0.0,
        'mean_sigma':0.0,
        'std_sigma':0.5,
        # to use a sigmoid for sigma, set the two following to diffrent than 0.0
        # Both are set as non-trainable in the probact.py file
        'alpha':0.0, # must not be integer
        'beta':0.0 # must not be integer
        }

        self.sigma = EWTrainableMuSigma([4 * hidden_size],**init_params)

        #self.sigma_ih = EWTrainableMuSigma([4 * hidden_size, num_features],**init_params)
        #self.sigma_hh = EWTrainableMuSigma([4 * hidden_size, hidden_size],**init_params)

        #self.sigma_ih = Parameter(torch.zeros_like(self.weight_ih))
        #self.sigma_hh = Parameter(torch.zeros_like(self.weight_hh))

    @jit.script_method
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
        #print(self.sigma(gates).size())
        #print(self.sigma(gates).mean())
        ingate, forgetgate, cellgate, outgate = gates_NAME.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        
        return hy, (hy, cy)


class LSTMCell(jit.ScriptModule):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, num_features))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
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


class LayerNorm(jit.ScriptModule):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        # XXX: This is true for our LSTM / NLP use case and helps simplify code
        assert len(normalized_shape) == 1

        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    @jit.script_method
    def compute_layernorm_stats(self, inp):
        mu = inp.mean(-1, keepdim=True)
        sigma = inp.std(-1, keepdim=True, unbiased=False)
        return mu, sigma

    @jit.script_method
    def forward(self, inp):
        mu, sigma = self.compute_layernorm_stats(inp)
        return (inp - mu) / sigma * self.weight + self.bias


class LayerNormLSTMCell(jit.ScriptModule):
    def __init__(self, num_features, hidden_size, decompose_layernorm=False):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, num_features))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases

        if decompose_layernorm:
            ln = LayerNorm
        else:
            ln = nn.LayerNorm

        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    @jit.script_method
    def forward(
        self, inp: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(inp, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(
        self, inp: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = inp.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class ReverseLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super().__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(
        self, inp: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = reverse(inp.unbind(0))
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(reverse(outputs)), state


class BidirLSTMLayer(jit.ScriptModule):
    __constants__ = ["directions"]

    def __init__(self, cell, *cell_args):
        super().__init__()
        self.directions = nn.ModuleList(
            [
                LSTMLayer(cell, *cell_args),
                ReverseLSTMLayer(cell, *cell_args),
            ]
        )

    @jit.script_method
    def forward(
        self, inp: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = jit.annotate(List[Tensor], [])
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(inp, state)
            outputs += [out]
            output_states += [out_state]
            i += 1
        return torch.cat(outputs, -1), output_states


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [
        layer(*other_layer_args) for _ in range(num_layers - 1)
    ]
    return nn.ModuleList(layers)


class StackedLSTM(jit.ScriptModule):
    __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    @jit.script_method
    def forward(
        self, inp: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = inp
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


# Differs from StackedLSTM in that its forward method takes
# List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass StackedLSTM
# except we don't support overriding script methods.
# https://github.com/pytorch/pytorch/issues/10733
class StackedLSTM2(jit.ScriptModule):
    __constants__ = ["layers"]  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )

    @jit.script_method
    def forward(
        self, inp: Tensor, states: List[List[Tuple[Tensor, Tensor]]]
    ) -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]:
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = jit.annotate(List[List[Tuple[Tensor, Tensor]]], [])
        output = inp
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


class StackedLSTMWithDropout(jit.ScriptModule):
    # Necessary for iterating through self.layers and dropout support
    __constants__ = ["layers", "num_layers"]

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super().__init__()
        self.layers = init_stacked_lstm(
            num_layers, layer, first_layer_args, other_layer_args
        )
        # Introduces a Dropout layer on the outputs of each LSTM layer except
        # the last layer, with dropout probability = 0.4.
        self.num_layers = num_layers

        if num_layers == 1:
            warnings.warn(
                "dropout lstm adds dropout layers after all but last "
                "recurrent layer, it expects num_layers greater than "
                "1, but got num_layers = 1"
            )

        self.dropout_layer = nn.Dropout(0.4)

    @jit.script_method
    def forward(
        self, inp: Tensor, states: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
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


class MyStackedLSTMWithDropout(jit.ScriptModule):
    # Necessary for iterating through self.layers and dropout support
    __constants__ = ["layers", "num_layers"]

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
        print('dropout',dropout)
        self.dropout_layer = nn.Dropout(dropout)

    @jit.script_method
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
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
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


def flatten_states(states):
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]


def double_flatten_states(states):
    # XXX: Can probably write this in a nicer way
    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]


def test_script_rnn_layer(seq_len, batch, num_features, hidden_size):
    inp = torch.randn(seq_len, batch, num_features)
    state = LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
    rnn = LSTMLayer(LSTMCell, num_features, hidden_size)
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(num_features, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_rnn(seq_len, batch, num_features, hidden_size, num_layers):
    inp = torch.randn(seq_len, batch, num_features)
    states = [
        LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
        for _ in range(num_layers)
    ]
    rnn = script_lstm(num_features, hidden_size, num_layers)
    out, out_state = rnn(inp, states)
    custom_state = flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(num_features, hidden_size, num_layers)
    lstm_state = flatten_states(states)
    for layer in range(num_layers):
        custom_params = list(rnn.parameters())[4 * layer : 4 * (layer + 1)]
        for lstm_param, custom_param in zip(lstm.all_weights[layer], custom_params):
            assert lstm_param.shape == custom_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_bidir_rnn(seq_len, batch, num_features, hidden_size, num_layers):
    inp = torch.randn(seq_len, batch, num_features)
    states = [
        [
            LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
            for _ in range(2)
        ]
        for _ in range(num_layers)
    ]
    rnn = script_lstm(num_features, hidden_size, num_layers, bidirectional=True)
    out, out_state = rnn(inp, states)
    custom_state = double_flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(num_features, hidden_size, num_layers, bidirectional=True)
    lstm_state = double_flatten_states(states)
    for layer in range(num_layers):
        for direct in range(2):
            index = 2 * layer + direct
            custom_params = list(rnn.parameters())[4 * index : 4 * index + 4]
            for lstm_param, custom_param in zip(lstm.all_weights[index], custom_params):
                assert lstm_param.shape == custom_param.shape
                with torch.no_grad():
                    lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_lstm_dropout(
    seq_len, batch, num_features, hidden_size, num_layers
):
    inp = torch.randn(seq_len, batch, num_features)
    # Recall the definition above in the code: LSTMState = namedtuple("LSTMState", ["hx", "cx"])
    states = [
        LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
        for _ in range(num_layers)
    ]
    rnn = script_lstm(num_features, hidden_size, num_layers, dropout=True)

    # just a smoke test
    out, out_state = rnn(inp, states)


def test_script_stacked_lnlstm(seq_len, batch, num_features, hidden_size, num_layers):
    inp = torch.randn(seq_len, batch, num_features)
    states = [
        LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size))
        for _ in range(num_layers)
    ]
    rnn = script_lnlstm(num_features, hidden_size, num_layers)

    # just a smoke test
    out, out_state = rnn(inp, states)


test_script_rnn_layer(5, 2, 3, 7)
test_script_stacked_rnn(5, 2, 3, 7, 4)
test_script_stacked_bidir_rnn(5, 2, 3, 7, 4)
test_script_stacked_lstm_dropout(5, 2, 3, 7, 4)
test_script_stacked_lnlstm(5, 2, 3, 7, 4)
