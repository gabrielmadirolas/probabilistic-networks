"""Custom LSTMs with configurable probabilistic gate perturbations.

The custom cell is intentional: it exposes the four gate pre-activations so
experiments can apply probabilistic perturbations before or after their
nonlinearities. The explicit time-step loop is flexible, but is generally
slower than PyTorch's fused ``nn.LSTM`` kernels for long sequences.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.probact import ProbabilisticActivation

State = Tuple[Tensor, Tensor]


def _probability_module(
    hidden_size: int,
    prob_params: Optional[dict],
    *,
    sharing: str,
    distribution: str,
    location: str,
    sample_mode: str,
    train_mu: Optional[bool],
    train_sigma: bool,
) -> ProbabilisticActivation:
    params = prob_params or {}
    if train_mu is None:
        train_mu = bool(params.get("std_mu", 0.0))
    alpha, beta = params.get("alpha"), params.get("beta")
    if not alpha or not beta:
        alpha = beta = None
    return ProbabilisticActivation(
        4 * hidden_size,
        sharing=sharing,
        mean=float(params.get("mean_mu", 0.0)),
        mean_std=float(params.get("std_mu", 0.0)),
        scale=float(params.get("mean_sigma", 0.0)),
        scale_std=float(params.get("std_sigma", 0.0)),
        train_mean=train_mu,
        train_scale=train_sigma,
        distribution=distribution,
        location=location,
        sample_mode=sample_mode,
        alpha=alpha,
        beta=beta,
    )


def prob_lstm(
    num_features: int,
    hidden_size: int,
    num_layers: int,
    batch_first: bool = False,
    dropout: float = 0.1,
    prob_params: Optional[dict] = None,
    *,
    sharing: str = "per_node",
    distribution: str = "normal",
    noise_location: str = "pre_activation",
    sample_mode: str = "train_only",
    train_mu: Optional[bool] = None,
    train_sigma: bool = True,
    bidirectional: bool = False,
    proj_size: int = 0,
) -> "MyStackedLSTMWithDropout":
    """Build a probabilistic LSTM.

    ``sharing`` controls learned mean/scale granularity: ``per_node`` learns
    one value for every gate/node channel and direction, ``per_layer`` shares
    one value within each recurrent layer, and ``per_network`` shares one
    value across the complete recurrent stack. ``sample_mode`` is ``always``,
    ``train_only`` (the deterministic-evaluation default), or ``never``.

    Inputs and outputs follow ``batch_first``. Optional initial states use
    ``(h_0, c_0)`` with shapes ``(layers * directions, batch, output_size)``
    and ``(layers * directions, batch, hidden_size)`` respectively. The
    returned final state is ``(h_n, c_n)`` with those same conventions.
    """
    return MyStackedLSTMWithDropout(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        dropout=dropout,
        probabilistic=True,
        prob_params=prob_params,
        sharing=sharing,
        distribution=distribution,
        noise_location=noise_location,
        sample_mode=sample_mode,
        train_mu=train_mu,
        train_sigma=train_sigma,
        bidirectional=bidirectional,
        proj_size=proj_size,
    )


def my_lstm(
    num_features: int,
    hidden_size: int,
    num_layers: int,
    batch_first: bool = False,
    dropout: float = 0.1,
    *,
    bidirectional: bool = False,
    proj_size: int = 0,
) -> "MyStackedLSTMWithDropout":
    """Build the non-probabilistic custom LSTM baseline."""
    return MyStackedLSTMWithDropout(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        dropout=dropout,
        probabilistic=False,
        bidirectional=bidirectional,
        proj_size=proj_size,
    )


class ProbLSTMCell(nn.Module):
    """An LSTM cell with configurable noise on the four gate channels."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        prob_module: ProbabilisticActivation,
        proj_size: int = 0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.output_size = proj_size or hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, num_features))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, self.output_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        self.weight_hr = (
            nn.Parameter(torch.empty(self.output_size, hidden_size))
            if proj_size
            else None
        )
        # Keep this module name so existing checkpoints retain
        # ``...cell.sigma.sigma`` and ``...cell.sigma.mu`` keys.
        self.sigma = prob_module
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bound = self.hidden_size**-0.5
        for parameter in (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh):
            nn.init.uniform_(parameter, -bound, bound)
        with torch.no_grad():
            self.bias_ih[self.hidden_size : 2 * self.hidden_size].fill_(1.0)
        if self.weight_hr is not None:
            nn.init.uniform_(self.weight_hr, -bound, bound)

    def forward(
        self,
        inp: Tensor,
        state: State,
        *,
        probabilistic: bool = True,
        input_gates: Optional[Tensor] = None,
    ) -> Tuple[Tensor, State]:
        hx, cx = state
        if input_gates is None:
            input_gates = torch.mm(inp, self.weight_ih.t()) + self.bias_ih
        gates = input_gates + torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        if probabilistic and self.sigma.location == "pre_activation":
            gates = self.sigma(gates)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        activated = torch.cat((ingate, forgetgate, cellgate, outgate), dim=1)
        if probabilistic and self.sigma.location == "post_activation":
            activated = self.sigma(activated)
            ingate, forgetgate, cellgate, outgate = activated.chunk(4, dim=1)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        if self.weight_hr is not None:
            hy = torch.mm(hy, self.weight_hr.t())
        return hy, (hy, cy)


class LSTMCell(nn.Module):
    """Non-probabilistic LSTM cell with the same parameterization."""

    def __init__(self, num_features: int, hidden_size: int, proj_size: int = 0) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.output_size = proj_size or hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, num_features))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, self.output_size))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        self.weight_hr = (
            nn.Parameter(torch.empty(self.output_size, hidden_size))
            if proj_size
            else None
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        bound = self.hidden_size**-0.5
        for parameter in (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh):
            nn.init.uniform_(parameter, -bound, bound)
        with torch.no_grad():
            self.bias_ih[self.hidden_size : 2 * self.hidden_size].fill_(1.0)
        if self.weight_hr is not None:
            nn.init.uniform_(self.weight_hr, -bound, bound)

    def forward(
        self, inp: Tensor, state: State, input_gates: Optional[Tensor] = None
    ) -> Tuple[Tensor, State]:
        hx, cx = state
        if input_gates is None:
            input_gates = torch.mm(inp, self.weight_ih.t()) + self.bias_ih
        gates = input_gates + torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        if self.weight_hr is not None:
            hy = torch.mm(hy, self.weight_hr.t())
        return hy, (hy, cy)


class LSTMLayer(nn.Module):
    """Run one or two directional cells across a sequence."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        probabilistic: bool,
        proj_size: int,
        bidirectional: bool,
        prob_modules: Optional[List[ProbabilisticActivation]],
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        cell_type = ProbLSTMCell if probabilistic else LSTMCell
        prob_forward = prob_modules[0] if probabilistic else None
        self.cell = (
            cell_type(input_size, hidden_size, prob_forward, proj_size)
            if probabilistic
            else cell_type(input_size, hidden_size, proj_size)
        )
        if bidirectional:
            prob_reverse = prob_modules[1] if probabilistic else None
            self.reverse_cell = (
                cell_type(input_size, hidden_size, prob_reverse, proj_size)
                if probabilistic
                else cell_type(input_size, hidden_size, proj_size)
            )
        else:
            self.reverse_cell = None
        self.probabilistic = probabilistic

    def _run_direction(self, inp: Tensor, state: State, cell: nn.Module, reverse=False):
        # The input projection is independent of recurrent state, so calculate
        # it once for all time steps instead of launching a matrix multiply per
        # step. The recurrent projection remains inside the sequential loop.
        input_gates = torch.nn.functional.linear(inp, cell.weight_ih, cell.bias_ih)
        steps = tuple(zip(inp.unbind(0), input_gates.unbind(0)))
        if reverse:
            steps = tuple(reversed(steps))
        outputs = []
        for step, input_gate in steps:
            if self.probabilistic:
                output, state = cell(
                    step, state, probabilistic=True, input_gates=input_gate
                )
            else:
                output, state = cell(step, state, input_gates=input_gate)
            outputs.append(output)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=0), state

    def forward(self, inp: Tensor, states: List[State]):
        forward_output, forward_state = self._run_direction(
            inp, states[0], self.cell
        )
        if self.reverse_cell is None:
            return forward_output, [forward_state]
        reverse_output, reverse_state = self._run_direction(
            inp, states[1], self.reverse_cell, reverse=True
        )
        return torch.cat((forward_output, reverse_output), dim=2), [
            forward_state,
            reverse_state,
        ]


class MyStackedLSTMWithDropout(nn.Module):
    """Stack custom LSTM layers with PyTorch-like input, output, and states."""

    def __init__(
        self,
        *,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        dropout: float,
        probabilistic: bool,
        prob_params: Optional[dict] = None,
        sharing: str = "per_node",
        distribution: str = "normal",
        noise_location: str = "pre_activation",
        sample_mode: str = "train_only",
        train_mu: Optional[bool] = None,
        train_sigma: bool = True,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        super().__init__()
        if num_features <= 0 or hidden_size <= 0:
            raise ValueError("num_features and hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be at least 1")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if proj_size < 0 or proj_size >= hidden_size:
            raise ValueError("proj_size must be zero or smaller than hidden_size")
        if not probabilistic and sharing != "per_node":
            raise ValueError("sharing only applies when probabilistic=True")

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.output_size = proj_size or hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.probabilistic = probabilistic
        self.proj_size = proj_size
        self.dropout = nn.Dropout(dropout) if num_layers > 1 else nn.Identity()

        network_prob_module = None
        if probabilistic and sharing == "per_network":
            network_prob_module = _probability_module(
                hidden_size,
                prob_params,
                sharing="per_network",
                distribution=distribution,
                location=noise_location,
                sample_mode=sample_mode,
                train_mu=train_mu,
                train_sigma=train_sigma,
            )

        layers = []
        layer_input_size = num_features
        for _ in range(num_layers):
            prob_modules = None
            if probabilistic and sharing == "per_layer":
                layer_prob_module = _probability_module(
                    hidden_size,
                    prob_params,
                    sharing="per_layer",
                    distribution=distribution,
                    location=noise_location,
                    sample_mode=sample_mode,
                    train_mu=train_mu,
                    train_sigma=train_sigma,
                )
                prob_modules = [layer_prob_module] * self.num_directions
            elif probabilistic and sharing == "per_network":
                prob_modules = [network_prob_module] * self.num_directions
            elif probabilistic:
                prob_modules = [
                    _probability_module(
                        hidden_size,
                        prob_params,
                        sharing="per_node",
                        distribution=distribution,
                        location=noise_location,
                        sample_mode=sample_mode,
                        train_mu=train_mu,
                        train_sigma=train_sigma,
                    )
                    for _ in range(self.num_directions)
                ]

            layer = LSTMLayer(
                layer_input_size,
                hidden_size,
                probabilistic=probabilistic,
                proj_size=proj_size,
                bidirectional=bidirectional,
                prob_modules=prob_modules,
            )
            layers.append(layer)
            layer_input_size = self.num_directions * self.output_size

        self.layers = nn.ModuleList(layers)

    def _zero_state(self, inp: Tensor, batch: int) -> State:
        h0 = inp.new_zeros(
            self.num_layers * self.num_directions, batch, self.output_size
        )
        c0 = inp.new_zeros(
            self.num_layers * self.num_directions, batch, self.hidden_size
        )
        return h0, c0

    def _validate_state(self, state: State, batch: int) -> None:
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise TypeError("state must be a pair (h_0, c_0)")
        expected_h = (self.num_layers * self.num_directions, batch, self.output_size)
        expected_c = (self.num_layers * self.num_directions, batch, self.hidden_size)
        if tuple(state[0].shape) != expected_h or tuple(state[1].shape) != expected_c:
            raise ValueError(
                f"state shapes must be {expected_h} and {expected_c}; got "
                f"{tuple(state[0].shape)} and {tuple(state[1].shape)}"
            )

    def forward(self, inp: Tensor, state: Optional[State] = None):
        if inp.ndim != 3:
            raise ValueError("input must have three dimensions: batch, sequence, features")
        batch_size = inp.shape[0] if self.batch_first else inp.shape[1]
        if self.batch_first:
            inp = inp.transpose(0, 1)
        if inp.shape[2] != self.num_features:
            raise ValueError(
                f"input feature dimension must be {self.num_features}; got {inp.shape[2]}"
            )

        sequence_length = inp.shape[0]
        if sequence_length == 0:
            raise ValueError("input sequence must contain at least one time step")
        state = self._zero_state(inp, batch_size) if state is None else state
        self._validate_state(state, batch_size)
        if any(t.device != inp.device or t.dtype != inp.dtype for t in state):
            raise ValueError("initial states must match the input device and dtype")

        h0, c0 = state
        output = inp
        final_h: List[Tensor] = []
        final_c: List[Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            layer_states: List[State] = []
            for direction in range(self.num_directions):
                state_index = layer_index * self.num_directions + direction
                layer_states.append((h0[state_index], c0[state_index]))
            output, states = layer(output, layer_states)
            for hidden, cell in states:
                final_h.append(hidden)
                final_c.append(cell)
            if layer_index < self.num_layers - 1:
                output = self.dropout(output)

        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (torch.stack(final_h), torch.stack(final_c))
