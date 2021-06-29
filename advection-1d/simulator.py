#!/usr/bin/env python

import functools

import numpy as np
import matplotlib.pyplot as plt

from pde_superresolution import integrate
from pde_superresolution import duckarray
from pde_superresolution import equations

from constants import TRUE_AMPLITUDE, TIME_MAX, TIME_DELTA, SEED


class RandomForcing(object):
    """Deterministic random forcing, periodic in both space and time."""

    def __init__(self, grid, nparams=20, seed=0, amplitude=1, k_min=1, k_max=1, phi=0):
        self.grid = grid
        rs = np.random.RandomState(seed)
        self.a = 0.5 * amplitude * rs.uniform(-1, 1, size=(nparams, 1))
        self.omega = rs.uniform(-0.4, 0.4, size=(nparams, 1))
        k_values = np.arange(k_min, k_max + 1)
        self.k = rs.choice(np.concatenate([-k_values, k_values]), size=(nparams, 1))
        # self.phi = rs.uniform(0, 2 * np.pi, size=(nparams, 1))
        self.phi = phi * np.ones(shape=(nparams, 1))

    def __call__(self, t):
        spatial_phase = 2 * np.pi * self.k * self.grid.reference_x / self.grid.period
        signals = duckarray.sin(self.omega * t + spatial_phase + self.phi)
        reference_forcing = duckarray.sum(self.a * signals, axis=0)
        return self.grid.resample(reference_forcing)


def run(amplitude=0, phi=0):
    equation = equations.EQUATION_TYPES["burgers"](
        random_seed=SEED, num_points=32, eta=0
    )
    forcing = RandomForcing(
        grid=equation.grid,
        nparams=1,
        seed=SEED,
        amplitude=amplitude,
        k_min=1,
        k_max=1,
        phi=phi,
    )
    forcing.omega = 0  # disable time dependency
    equation.forcing = forcing

    results = integrate.integrate_baseline(
        equation=equation,
        times=np.arange(0, TIME_MAX, TIME_DELTA),
        warmup=0,
        accuracy_order=1,
        integrate_method="RK23",
        exact_filter_interval=None,
    )

    x = results["x"].values
    y = results["y"].values
    time = results["time"].values
    y_simulation = y[1:4, :]

    return y_simulation, x


if __name__ == "__main__":
    y = run(amplitude=TRUE_AMPLITUDE)
    print(y)
