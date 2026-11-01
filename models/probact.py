"""Probabilistic activation and additive-noise modules.

The recurrent model uses ``ProbabilisticActivation`` to add a learned
Gaussian offset to LSTM gate values.  The module can also be used around an
ordinary activation function, which makes the same implementation useful for
experiments outside the custom LSTM.
"""

from __future__ import annotations

from numbers import Number
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


SharingScope = str
Shape = Union[int, Sequence[int], torch.Size]
_SCALE_EPSILON = 1e-6


def _inverse_softplus(value: Tensor) -> Tensor:
    """Map nonnegative scales to a stable unconstrained parameter space."""
    value = value.clamp_min(_SCALE_EPSILON)
    return value + torch.log(-torch.expm1(-value))


class _LegacyScaleLoading:
    """Convert pre-softplus ``sigma`` values when loading old checkpoints."""

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version_key = prefix + "scale_parameterization_version"
        sigma_key = prefix + "sigma"
        if version_key not in state_dict and sigma_key in state_dict:
            old_scale = state_dict[sigma_key].abs()
            if self._train_scale:
                old_scale = _inverse_softplus(old_scale)
            state_dict[sigma_key] = old_scale
            state_dict[version_key] = self.scale_parameterization_version.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _shape_tuple(shape: Shape) -> tuple[int, ...]:
    if isinstance(shape, int):
        shape = (shape,)
    result = tuple(int(size) for size in shape)
    if not result or any(size <= 0 for size in result):
        raise ValueError("num_parameters must contain positive dimensions")
    return result


class ProbabilisticActivation(_LegacyScaleLoading, nn.Module):
    """Add learnable Gaussian or variance-matched uniform noise to ``x``.

    Args:
        num_parameters: Shape of the per-node parameters. An LSTM supplies
            ``4 * hidden_size`` entries, one for each gate/node channel.
        sharing: ``'per_node'`` creates one value per entry;
            ``'per_layer'`` creates one value per module (one module per
            recurrent layer); ``'per_network'`` creates one value per module
            and the caller shares that module across recurrent layers.
        mean: Initial additive mean offset.
        scale: Initial standard deviation. A trainable scale is represented by
            an unconstrained raw value and mapped through softplus, so it stays
            positive and remains trainable when initialized at zero.
        train_mean: Whether the mean offset is learned. A fixed mean is a
            Python scalar and adds no checkpoint entry.
        train_scale: Whether the standard deviation is learned.
        distribution: ``'normal'`` or variance-matched ``'uniform'`` noise.
        location: ``'pre_activation'`` or ``'post_activation'``. When an
            activation is supplied to :meth:`forward`, this selects where the
            noise is inserted. Without an activation, noise is added to ``x``.
        sample_mode: ``'always'`` samples in train and eval modes;
            ``'train_only'`` samples only while training; ``'never'`` uses
            the learned mean offset without a random draw.
        alpha, beta: Optional fixed parameters for the bounded scale
            ``alpha * sigmoid(beta * scale)``. When omitted, scale is
            unbounded above.

    ``per_network`` sharing is achieved by passing the same module instance
    to each recurrent layer. The module itself does not discover its parent
    model or create hidden global state.
    """

    VALID_SHARING = {"per_node", "per_layer", "per_network"}
    VALID_DISTRIBUTIONS = {"normal", "uniform"}
    VALID_LOCATIONS = {"pre_activation", "post_activation"}
    VALID_SAMPLE_MODES = {"always", "train_only", "never"}

    def __init__(
        self,
        num_parameters: Shape,
        *,
        sharing: SharingScope = "per_node",
        mean: float = 0.0,
        mean_std: float = 0.0,
        scale: float = 0.0,
        scale_std: float = 0.0,
        train_mean: bool = False,
        train_scale: bool = True,
        distribution: str = "normal",
        location: str = "pre_activation",
        sample_mode: str = "train_only",
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        super().__init__()
        if sharing not in self.VALID_SHARING:
            raise ValueError(f"sharing must be one of {sorted(self.VALID_SHARING)}")
        if distribution not in self.VALID_DISTRIBUTIONS:
            raise ValueError(
                f"distribution must be one of {sorted(self.VALID_DISTRIBUTIONS)}"
            )
        if location not in self.VALID_LOCATIONS:
            raise ValueError(f"location must be one of {sorted(self.VALID_LOCATIONS)}")
        if sample_mode not in self.VALID_SAMPLE_MODES:
            raise ValueError(
                f"sample_mode must be one of {sorted(self.VALID_SAMPLE_MODES)}"
            )
        if not isinstance(mean, Number) or not isinstance(scale, Number):
            raise TypeError("mean and scale initializers must be numeric scalars")
        if mean_std < 0 or scale < 0 or scale_std < 0:
            raise ValueError("mean and scale initializer deviations must be nonnegative")
        if (alpha is None) != (beta is None):
            raise ValueError("alpha and beta must either both be set or both be None")

        self.sharing = sharing
        self.distribution = distribution
        self.location = location
        self.sample_mode = sample_mode
        self.alpha = alpha
        self.beta = beta
        self._train_scale = train_scale
        self.parameter_shape = _shape_tuple(num_parameters)
        parameter_shape = self.parameter_shape if sharing == "per_node" else (1,)

        # Register mean before scale to preserve the legacy optimizer parameter
        # order when both were trainable.
        if train_mean:
            self.mu = nn.Parameter(
                torch.empty(parameter_shape).normal_(
                    mean=float(mean), std=float(mean_std)
                )
            )
        else:
            self.mu = float(mean)

        # Preserve the legacy parameter name ``sigma`` for checkpoint loading.
        initial_scale = torch.empty(parameter_shape).normal_(
            mean=float(scale), std=float(scale_std)
        ).abs_()
        if train_scale:
            self.sigma = nn.Parameter(_inverse_softplus(initial_scale))
        else:
            self.register_buffer("sigma", initial_scale)
        self.register_buffer(
            "scale_parameterization_version", torch.tensor(1, dtype=torch.uint8)
        )

    def effective_scale(self, reference: Tensor) -> Tensor:
        if self._train_scale:
            scale = F.softplus(self.sigma)
        else:
            scale = self.sigma
        scale = scale.to(device=reference.device, dtype=reference.dtype)
        if self.alpha is not None and self.beta is not None:
            scale = self.alpha * torch.sigmoid(self.beta * scale)
        return scale

    def noise(self, reference: Tensor) -> Tensor:
        """Return the additive offset, using ``reference`` for shape/device/dtype."""
        if isinstance(self.mu, Tensor):
            mean = self.mu.to(device=reference.device, dtype=reference.dtype)
        else:
            mean = reference.new_tensor(self.mu)

        should_sample = self.sample_mode == "always" or (
            self.sample_mode == "train_only" and self.training
        )
        if not should_sample:
            return mean

        if self.distribution == "normal":
            epsilon = torch.randn_like(reference)
        else:
            # Uniform[-sqrt(3), sqrt(3)] has unit variance.
            epsilon = (torch.rand_like(reference) * 2.0 - 1.0) * (3.0**0.5)
        return mean + self.effective_scale(reference) * epsilon

    def forward(
        self, x: Tensor, activation: Optional[Callable[[Tensor], Tensor]] = None
    ) -> Tensor:
        if activation is None:
            return x + self.noise(x)
        if self.location == "pre_activation":
            return activation(x + self.noise(x))
        return activation(x) + self.noise(x)


class EWTrainableMuSigma(ProbabilisticActivation):
    """Backward-compatible interface for the original element-wise module.

    The old module returned only the additive offset, because callers added
    that result to their pre-activation values themselves. It also keeps the
    historical ``mu``/``sigma`` state-dictionary names.
    """

    def __init__(self, num_parameters: Shape, prob_params: dict) -> None:
        mean_std = float(prob_params.get("std_mu", 0.0))
        alpha, beta = prob_params.get("alpha"), prob_params.get("beta")
        if not alpha or not beta:
            alpha = beta = None
        super().__init__(
            num_parameters,
            sharing="per_node",
            mean=float(prob_params.get("mean_mu", 0.0)),
            mean_std=mean_std,
            scale=float(prob_params.get("mean_sigma", 0.0)),
            train_mean=bool(mean_std),
            train_scale=True,
            distribution="normal",
            location="pre_activation",
            sample_mode="always",
            alpha=alpha,
            beta=beta,
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.noise(x)


class TrainableSigma(_LegacyScaleLoading, nn.Module):
    """Legacy ReLU-plus-noise activation with device/dtype-safe sampling."""

    def __init__(self, num_parameters: int = 1, init_sigma: float = 0.0) -> None:
        super().__init__()
        self._train_scale = True
        self.sigma = nn.Parameter(
            _inverse_softplus(torch.full((num_parameters,), float(init_sigma)))
        )
        self.register_buffer(
            "scale_parameterization_version", torch.tensor(1, dtype=torch.uint8)
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x) + F.softplus(self.sigma).to(x) * torch.randn_like(x)


class TrainableMuSigma(_LegacyScaleLoading, nn.Module):
    """Legacy ReLU-plus-noise activation with a learned scalar offset."""

    def __init__(self, init_mu: float = 0.0, init_sigma: float = 0.0) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(float(init_mu)))
        self._train_scale = True
        self.sigma = nn.Parameter(
            _inverse_softplus(torch.tensor(float(init_sigma)))
        )
        self.register_buffer(
            "scale_parameterization_version", torch.tensor(1, dtype=torch.uint8)
        )

    def forward(self, x: Tensor) -> Tensor:
        return (
            self.mu.to(x)
            + F.relu(x)
            + F.softplus(self.sigma).to(x) * torch.randn_like(x)
        )
