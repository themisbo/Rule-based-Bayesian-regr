#!/usr/bin/env burgers

import os
import sys

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal

from datadrivenpdes.core import equations
from datadrivenpdes.core import grids
import datadrivenpdes as pde

tf.enable_eager_execution()
import xarray as xr


def run(SIZE=24, dif=0.02, vel=-0.1, start_x=5, start_y=5):

    equation = pde.advection.equations.FiniteVolumeAdvectionDiffusion(
        diffusion_coefficient=dif
    )
    grid = grids.Grid.from_period(size=SIZE, length=2 * np.pi)
    x, y = grid.get_mesh()

    initial_state = equation.random_state(grid, seed=7109179)

    initial_state["x_velocity"] = vel * np.ones(shape=(SIZE, SIZE), dtype=np.float32)
    initial_state["y_velocity"] = vel * np.ones(shape=(SIZE, SIZE), dtype=np.float32)

    init_dist = multivariate_normal(mean=[start_x, start_y], cov=[[0.1, 0], [0, 0.1]])
    init = np.zeros((SIZE, SIZE), dtype=np.float32)
    for i in range(x[:, 0].size):
        for j in range(y[0, :].size):
            init[i, j] = init_dist.pdf([x[i, 0], y[0, j]])

    initial_state["concentration"] = init

    time_step = equation.get_time_step(grid)
    times = time_step * np.arange(150)
    results = pde.core.integrate.integrate_times(
        model=pde.core.models.FiniteDifferenceModel(equation, grid),
        state=initial_state,
        times=times,
        axis=0,
    )

    conc = results["concentration"].numpy()

    return conc, x[:, 0], y[0], times
