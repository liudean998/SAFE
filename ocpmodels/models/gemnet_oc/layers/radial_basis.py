"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import Dict, Union

import numpy as np
import torch
from scipy.special import binom
from scipy.special import sph_harm

from ocpmodels.common.typing import assert_is_instance
from ocpmodels.modules.scaling import ScaleFactor
from e3nn.o3 import spherical_harmonics, Irreps
from .base_layers import Dense


class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.

    Arguments
    ---------
        exponent: int
            Exponent of the envelope function.
    """

    def __init__(self, exponent: int) -> None:
        super().__init__()
        assert exponent > 0
        self.p = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = (
            1
            + self.a * d_scaled**self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))


class ExponentialEnvelope(torch.nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = torch.exp(
            -(d_scaled**2) / ((1 - d_scaled) * (1 + d_scaled))
        )
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))

class TanhEnvelope(torch.nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        # env_val = torch.tanh(d_scaled)
        # env_val = torch.sigmoid(d_scaled)
        env_val = 0.3 + 0.7 * (d_scaled**2)
        return torch.where(d_scaled <= 1, env_val, torch.zeros_like(d_scaled))

class GaussianBasis(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        if trainable:
            self.offset = torch.nn.Parameter(offset, requires_grad=True)
        else:
            self.register_buffer("offset", offset)
        self.coeff = -0.5 / ((stop - start) / (num_gaussians - 1)) ** 2
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist[:, None] - self.offset[None, :]
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SphericalBesselBasis(torch.nn.Module):
    """
    First-order spherical Bessel basis

    Arguments
    ---------
    num_radial: int
        Number of basis functions. Controls the maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.norm_const = math.sqrt(2 / (cutoff**3))
        # cutoff ** 3 to counteract dividing by d_scaled = d / cutoff

        # Initialize frequencies at canonical positions
        self.frequencies = torch.nn.Parameter(
            data=torch.tensor(
                np.pi * np.arange(1, num_radial + 1, dtype=np.float32)
            ),
            requires_grad=True,
        )

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        return (
            self.norm_const
            / d_scaled[:, None]
            * torch.sin(self.frequencies * d_scaled[:, None])
        )  # (num_edges, num_radial)


class BernsteinBasis(torch.nn.Module):
    """
    Bernstein polynomial basis,
    as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects

    Arguments
    ---------
    num_radial: int
        Number of basis functions. Controls the maximum frequency.
    pregamma_initial: float
        Initial value of exponential coefficient gamma.
        Default: gamma = 0.5 * a_0**-1 = 0.94486,
        inverse softplus -> pregamma = log e**gamma - 1 = 0.45264
    """

    def __init__(
        self,
        num_radial: int,
        pregamma_initial: float = 0.45264,
    ) -> None:
        super().__init__()
        prefactor = binom(num_radial - 1, np.arange(num_radial))
        self.register_buffer(
            "prefactor",
            torch.tensor(prefactor, dtype=torch.float),
            persistent=False,
        )

        self.pregamma = torch.nn.Parameter(
            data=torch.tensor(pregamma_initial, dtype=torch.float),
            requires_grad=True,
        )
        self.softplus = torch.nn.Softplus()

        exp1 = torch.arange(num_radial)
        self.register_buffer("exp1", exp1[None, :], persistent=False)
        exp2 = num_radial - 1 - exp1
        self.register_buffer("exp2", exp2[None, :], persistent=False)

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        gamma = self.softplus(self.pregamma)  # constrain to positive
        exp_d = torch.exp(-gamma * d_scaled)[:, None]
        return (
            self.prefactor * (exp_d**self.exp1) * ((1 - exp_d) ** self.exp2)
        )


class RadialBasis(torch.nn.Module):
    """

    Arguments
    ---------
    num_radial: int
        Number of basis functions. Controls the maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    scale_basis: bool
        Whether to scale the basis values for better numerical stability.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: Dict[str, str] = {"name": "gaussian"},
        envelope: Dict[str, Union[str, int]] = {
            "name": "polynomial",
            "exponent": 5,
        },
        scale_basis: bool = False,
    ) -> None:
        super().__init__()
        self.inv_cutoff = 1 / cutoff

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_rbf = ScaleFactor()

        env_name = assert_is_instance(envelope["name"], str).lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]

        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == "gaussian":
            self.rbf = GaussianBasis(
                start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            )
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(
                num_radial=num_radial, cutoff=cutoff, **rbf_hparams
            )
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)
        res = env[:, None] * self.rbf(d_scaled)

        if self.scale_basis:
            res = self.scale_rbf(res)

        return res
        # (num_edges, num_radial) or (num_edges, num_orders * num_radial)


class RadialBasisVec(torch.nn.Module):
    """

    Arguments
    ---------
    num_radial: int
        Number of basis functions. Controls the maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    scale_basis: bool
        Whether to scale the basis values for better numerical stability.
    """

    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: Dict[str, str] = {"name": "gaussian"},
        envelope: Dict[str, Union[str, int]] = {
            "name": "polynomial",
            "exponent": 5,
        },
        scale_basis: bool = False,
        num_radial_v: int = 32
    ) -> None:
        super().__init__()
        # print(num_radial_v, 'oooooo')
        num_radial_all = num_radial
        # num_radial_v = 32
        num_radial_d = num_radial_all - 3*num_radial_v
        self.inv_cutoff = 1 / cutoff

        self.scale_basis = scale_basis
        if self.scale_basis:
            self.scale_rbf = ScaleFactor()

        env_name = assert_is_instance(envelope["name"], str).lower()
        env_hparams = envelope.copy()
        del env_hparams["name"]

        if env_name == "polynomial":
            self.envelope = PolynomialEnvelope(**env_hparams)
        elif env_name == "exponential":
            self.envelope = ExponentialEnvelope(**env_hparams)
        else:
            raise ValueError(f"Unknown envelope function '{env_name}'.")

        rbf_name = rbf["name"].lower()
        rbf_hparams = rbf.copy()
        del rbf_hparams["name"]

        # RBFs get distances scaled to be in [0, 1]
        if rbf_name == "gaussian":
            # self.rbf = GaussianBasis(
            #     start=0, stop=1, num_gaussians=num_radial, **rbf_hparams
            # )
            self.rbf = GaussianBasis(
                start=0, stop=1, num_gaussians=num_radial_d, **rbf_hparams
            )
            self.rbf_v = GaussianBasis(start=0, stop=1,
                                       num_gaussians=num_radial_v, **rbf_hparams)
        elif rbf_name == "spherical_bessel":
            self.rbf = SphericalBesselBasis(
                num_radial=num_radial, cutoff=cutoff, **rbf_hparams
            )
        elif rbf_name == "bernstein":
            self.rbf = BernsteinBasis(num_radial=num_radial, **rbf_hparams)
        else:
            raise ValueError(f"Unknown radial basis function '{rbf_name}'.")

    def forward(self, d: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)
        # print(env.size())
        res = env[:, None] * self.rbf(d_scaled)
        # print(res.size())
        if self.scale_basis:
            res = self.scale_rbf(res)
        v = v.view(-1)
        # print(v.size())
        v_scaled = v * self.inv_cutoff
        env_v = self.envelope(v_scaled)
        # print(env_v.size())
        res_v = env_v[:, None]*self.rbf_v(v_scaled)
        res_v = reshape_vector3(res_v)
        res_vd  = torch.cat((res_v, res), dim=1)
        return res_vd


def reshape_vector(vector):
    size = vector.shape
    vector = vector.view(int(size[0]/4),
                             size[1]*4)
    return vector

def reshape_vector3(vector):
    size = vector.shape
    # print(size)
    vector = vector.view(int(size[0]/3),
                             size[1]*3)
    return vector

