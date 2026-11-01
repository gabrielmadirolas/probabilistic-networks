# Configurable probabilistic LSTM design

## Intent

Make the custom recurrent model a useful research instrument for comparing probabilistic activation behavior and parameter-sharing choices while retaining the current airline forecasting workflow and checkpoints.

## Goals

- Separate recurrent architecture from probabilistic activation/noise configuration.
- Support Gaussian perturbations before or after gate nonlinearities and explicit evaluation sampling policies.
- Make trainable mean and scale sharing selectable as per gate-node, per layer, or network-wide.
- Preserve a non-probabilistic custom LSTM baseline and flexible layer/hidden-size/dropout architecture.
- Match the usual PyTorch LSTM conventions for `batch_first`, initial/final states, bidirectionality, and projection.
- Retain compatibility with old model state dictionaries and update the root notebooks and documentation.

## Non-goals

- Replacing the custom recurrent cell with `nn.LSTM`; gate-level experiments require access to pre-activation values.
- Adding new dependencies, datasets, or a broad experiment-management framework.
- Claiming fused-cuDNN performance for a Python time-step loop.

## Design

Keep a custom LSTM cell because the probabilistic perturbation must be inserted into the gate computation. Implement it as composable cell/layer/stack modules with explicit validated configuration. The probability module owns the Gaussian sampling and trainable mean/scale. `per_node` gives each of the four gates at each hidden node its own values, `per_layer` gives each recurrent layer one shared value, and `per_network` shares values across all recurrent layers. Mean and scale use the same selected sharing policy; each can independently be trainable. Learned scales use softplus so they stay positive and remain trainable from a zero initialization. Old model weights are converted on load; old optimizer moments are not reused because they correspond to the prior parameterization.

Support perturbation before the gate nonlinearities (the current behavior) and after them (an explicit experimental option), and policies to sample in training and evaluation, in training only, or never. Preserve the current training behavior by keeping Gaussian, pre-activation perturbation, per-node trainable scale, fixed zero mean, and sampling during training as notebook defaults. Evaluation becomes deterministic by default, with Monte Carlo evaluation available explicitly.

Expose PyTorch-like input and state behavior: `batch_first` controls both input and output layout; optional `(h_0, c_0)` are accepted; final states are returned as stacked `(h_n, c_n)` tensors. Add bidirectional and projected recurrent architecture options, validation for invalid dimensions/options, standard uniform LSTM initialization, and inter-layer dropout only when an inter-layer boundary exists.

Keep the legacy `EWTrainableMuSigma` constructor and state-dictionary names as a compatibility layer. The notebooks will continue loading old checkpoints, stop manually transposing model outputs, and explain probabilistic settings and model architecture. `analyze_model.ipynb` will continue to inspect the legacy key and tolerate model variants.

## Approaches considered

- Replace the implementation with `nn.LSTM`: faster, but does not expose the gate values needed by this project.
- Keep the current code and only clean up factories: least disruptive, but leaves activation behavior and sharing scope implicit.
- **Selected:** retain a custom cell and separate probability module behind an explicit configuration API. It supports the research goal while preserving current notebook workflows.

## Constraints and success criteria

- Use the repository's existing PyTorch dependency; do not add packages.
- Keep current defaults compatible with the notebook's present probabilistic experiment.
- Make output/state shapes and parameter-sharing semantics explicit in docstrings and README.
- Update the training, Colab, and analysis notebooks consistently.
- Preserve old checkpoint loading behavior, including Colab resume when model parameter structure is unchanged.
