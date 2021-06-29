#!/usr/bin/env python

from cycler import cycler

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import theano

from simulator import run

plt.style.use("ggplot")

np.random.seed(1000005)
scale = 0.002

y_true, x = run(amplitude=0.001, phi=np.pi)
y_true = y_true * 10
shape = y_true.shape
times = shape[0]

noise = np.random.normal(scale=scale, size=shape)
y = y_true + noise
y_flat = y.flatten()

N_KNOT = 10
knots = np.linspace(0, 2 * np.pi, N_KNOT)
N_MODEL_KNOTS = 5 * N_KNOT
basis_funcs = sp.interpolate.BSpline(knots, np.eye(N_MODEL_KNOTS), k=3)

colors = ["red", "green", "blue"]
fig = plt.figure(figsize=(7.5, 5))
ax = fig.gca()
for i in range(times):
    plt.plot(
        x, y_true[i, :], color=colors[i], label="Exact solution for t=" + str(i + 1)
    )
    plt.plot(x, y[i, :], "o", color=colors[i], label="Data for t=" + str(i + 1))
plt.axvline(x=np.pi, linestyle="--")
ax.set_ylabel("u")
ax.set_xlabel("x")
ax.legend()
# plt.savefig("plots/advection_1d_data.png")


class BasisFunc:
    def __init__(self):
        """
        Initialize the class
        """
        self.Bx = None
        self.Bx_ = None

    def create_basis(self, x):
        """
        Create the basis object
        """
        Bx = basis_funcs(x)
        Bx_ = theano.shared(Bx)

        self.Bx = Bx
        self.Bx_ = Bx_


class RuleBasisFunc(BasisFunc):
    def __init__(self):
        """
        Initialize the class
        """
        super().__init__()
        self.discr = None

    def create_discr(self, no_points, min, max):
        """
        Create discretization
        """
        self.discr = np.linspace(min, max, no_points)


no_points = 25
xlow = 0
xmid = np.pi
xhi = 2 * np.pi

data_base = BasisFunc()
data_base.create_basis(x)

rule_first = RuleBasisFunc()
rule_first.create_discr(no_points, xlow, xmid)
rule_first.create_basis(rule_first.discr)

rule_second = RuleBasisFunc()
rule_second.create_discr(no_points, xmid, xhi)
rule_second.create_basis(rule_second.discr)

rule_third = RuleBasisFunc()
rule_third.create_discr(2, xlow, xhi)
rule_third.create_basis(rule_third.discr)


def logp_rule(a0, σ_a, Δ_a):
    """
    Construct the rule penalty
    """

    a = a0 + (σ_a * Δ_a).cumsum(axis=0)

    points_r1 = rule_first.Bx_.dot(a).T.flatten()
    points_r2 = rule_second.Bx_.dot(a).T.flatten()
    points_r3 = rule_third.Bx_.dot(a).T.flatten()

    rule_log_lik = 0
    for i in range(no_points):
        rule_log_lik = rule_log_lik + pm.math.switch(pm.math.lt(points_r1[i], 0), 1, 0)
        rule_log_lik = rule_log_lik + pm.math.switch(pm.math.gt(points_r2[i], 0), 1, 0)
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.gt(pm.math.abs_(points_r3[0] - points_r3[1]), 0.001), 1, 0
        )
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.gt(pm.math.abs_(points_r3[2] - points_r3[3]), 0.001), 1, 0
        )
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.gt(pm.math.abs_(points_r3[4] - points_r3[5]), 0.001), 1, 0
        )
        for num in range(times - 1):
            rule_log_lik = rule_log_lik + pm.math.switch(
                pm.math.lt(
                    points_r1[i + (num + 1) * no_points], points_r1[i + num * no_points]
                ),
                1,
                0,
            )
            rule_log_lik = rule_log_lik + pm.math.switch(
                pm.math.gt(
                    points_r2[i + (num + 1) * no_points], points_r2[i + num * no_points]
                ),
                1,
                0,
            )

    rule_ratio = rule_log_lik / ((times + 3) * no_points)

    return pm.Beta.dist(alpha=1.0, beta=100.0).logp(rule_ratio)


use_rule = True

with pm.Model() as model:
    σ_a = pm.HalfCauchy("σ_a", 0.1, shape=times)
    a0 = pm.Normal("a0", 0.0, 0.1, shape=times)
    Δ_a = pm.Normal("Δ_a", 0.0, 5.0, shape=(N_MODEL_KNOTS, times))
    a = pm.Deterministic("a", a0 + (σ_a * Δ_a).cumsum(axis=0))

    res = data_base.Bx_.dot(a)
    res_fl = res.T.flatten()

    obs = pm.Normal("obs", res_fl, scale, observed=y_flat)
    if use_rule:
        LL_rule = pm.Potential("LL_rule", logp_rule(a0, σ_a, Δ_a))

with model:
    trace = pm.sample_smc(draws=10000)

plt.rc("axes", prop_cycle=cycler("color", ["r", "g", "b"]))
thin = 10
fig = plt.figure(figsize=(7.5, 5))
ax = fig.gca()
for j in range(trace["a"].shape[2]):
    a = trace["a"][0 * thin, :, j]
    yvals = data_base.Bx.dot(a)
    plt.plot(x, yvals, label="Posterior curves for t=" + str(j + 1))
for iter in range(int(trace["a"].shape[0] / thin)):
    for j in range(trace["a"].shape[2]):
        a = trace["a"][iter * thin, :, j]
        yvals = data_base.Bx.dot(a)
        plt.plot(x, yvals, alpha=0.1)
for j in range(times):
    plt.scatter(x, y[j, :], label="Data for t=" + str(j + 1))
    plt.plot(x, y_true[j, :], color="k")
plt.plot(x, y_true[j, :], color="k", label="Exact solutions")
if use_rule:
    plt.axvline(x=np.pi, linestyle="--")
ax.set_ylabel("u")
ax.set_xlabel("x")
ax.legend()

plt.savefig("plots/advection_1d_posterior" + "_rule" * use_rule + ".png")

import pickle

pickle.dump(trace["a"], open("rule_spline", "wb"))
trace = pickle.load(open("rule_spline", "rb"))

mean_post = np.mean(trace["a"], axis=0)
mean_post = np.mean(trace, axis=0)

for j in range(mean_post.shape[1]):
    a = mean_post[:, j]
    yvals = data_base.Bx.dot(a)
    plt.plot(x, yvals, label="Posterior curves for t=" + str(j + 1))

# y_pred = []
# for j in range(mean_post.shape[1]):
#     a = mean_post[:, j]
#     yvals = data_base.Bx.dot(a)
#     y_pred.append(yvals)
# y_pred = np.array(y_pred)

res = data_base.Bx.dot(mean_post)
res_fl = res.T.flatten()


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

mean_squared_error(res_fl, y_flat)

mean_absolute_error(res_fl, y_flat)

explained_variance_score(res_fl, y_flat)

r2_score(res_fl, y_flat)

norule_waic = pm.waic(trace, model)
norule_loo = pm.loo(trace, model)
