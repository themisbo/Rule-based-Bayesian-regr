#!/usr/bin/env python

from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano
import xarray as xr

from simulator import run


concentration_simulation, x, y, times = run(start_x=5)

conc_true = xr.DataArray(
    concentration_simulation,
    dims=["time", "x", "y"],
    coords={"time": times, "x": x, "y": y},
)

frames_check = [0, 25, 50, 75, 100, 125, 149]
conc_true[frames_check, :, :].plot(col="time", robust=True, aspect=1)
plt.savefig("plots/burgers2D_indicativeData.png")

con_tr = concentration_simulation.flatten()

con_inp = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
con_inp_t = np.transpose(np.array(np.meshgrid(x, y, times, indexing="ij"))).reshape(
    -1, 3
)
con_inp_tt = np.flip(con_inp_t, axis=1)


class FramePic:
    def __init__(self, x, y, times, frames):
        """
        Initialize the class
        """
        self.x = x
        self.y = y
        self.times = times
        self.frames = frames
        self.test_con = None
        self.test_inp = None
        self.test_full = None
        self.max_con = None
        self.max_inp_inter = None
        self.max_inp = None
        self.f1_inter = None
        self.x_inter = None
        self.y_inter = None
        self.x_newinter = None
        self.y_newinter = None

    def inputdata(self):
        """
        Get frame values and indexes and construct data array object
        """

        test_con = np.concatenate(
            [
                con_tr[fr * x.size * y.size : (fr + 1) * x.size * y.size]
                for fr in self.frames
            ]
        )
        test_inp = np.concatenate(
            [
                con_inp_tt[fr * x.size * y.size : (fr + 1) * x.size * y.size]
                for fr in self.frames
            ]
        )

        test_full = xr.DataArray(
            test_con.reshape(-1, x.size, y.size),
            dims=["time", "x", "y"],
            coords={
                "time": np.unique(test_inp[:, 0]),
                "x": np.unique(test_inp[:, 1]),
                "y": np.unique(test_inp[:, 2]),
            },
        )

        self.test_con = test_con
        self.test_inp = test_inp
        self.test_full = test_full

    def getcenter(self):
        """
        Find and plot the center of the blobs for the frames
        """

        max_con = np.array(
            [
                con_tr[fr * x.size * y.size : (fr + 1) * x.size * y.size].max()
                for fr in self.frames
            ]
        )
        max_inp_inter = np.array(
            [
                con_tr[fr * x.size * y.size : (fr + 1) * x.size * y.size].argmax()
                for fr in self.frames
            ]
        )
        frame_maxpic = [
            max_inp_inter[i] + x.size * y.size * i for i in range(self.frames.size)
        ]
        max_inp = self.test_inp[frame_maxpic, :]

        self.max_con = max_con
        self.max_inp_inter = max_inp_inter
        self.max_inp = max_inp

    def interpolate_frames(self):
        """
        Center (x-value) - linear interpolation
        """

        self.x_inter = self.max_inp[:, 0]
        self.y_inter = self.max_inp[:, 1]
        self.f1_inter = interpolate.interp1d(self.x_inter, self.y_inter)
        self.x_newinter = np.arange(0, times.max(), 0.1)
        self.y_newinter = self.f1_inter(self.x_newinter)


class MakeRule:
    def __init__(self, input_frames, rule_frames):
        """
        Initialize the class
        """
        self.input_frames = input_frames
        self.rule_frames = rule_frames
        self.rula = None
        self.size_rule_all = None
        self.id_mat = np.eye(
            input_frames.x.size * input_frames.y.size * input_frames.frames.size
        )
        self.id_mat_ = theano.shared(self.id_mat)

    def get_rule(self):
        """
        Extract the indexes of the rules
        """

        center_rule = self.input_frames.f1_inter(self.rule_frames.max_inp[:, 0])
        center_rule_inp = self.rule_frames.max_inp
        center_rule_inp[:, 2] = center_rule
        self.rula = center_rule_inp
        self.size_rule_all = self.rula.shape[0]


frames_data = np.array([0, 75, 149])
input_frames = FramePic(x, y, times, frames_data)
input_frames.inputdata()
input_frames.test_full.plot(col="time", robust=True, aspect=1)
plt.savefig("plots/burgers2D_inputframes.png")
plt.show()

input_frames.getcenter()
for i in range(input_frames.frames.size):
    input_frames.test_full[i].plot()
    plt.plot(
        input_frames.max_inp[i][1], input_frames.max_inp[i][2], color="r", marker="o"
    )
    plt.savefig("plots/burgers2D_inputframe_center_" + str(i) + ".png")
    plt.show()

input_frames.interpolate_frames()

frames_rule = np.arange(1, 149, 5)
rule_frames = FramePic(x, y, times, frames_rule)
rule_frames.inputdata()

rule_frames.getcenter()

plt.plot(
    input_frames.x_inter,
    input_frames.y_inter,
    "o",
    input_frames.x_newinter,
    input_frames.y_newinter,
    "-",
)
plt.plot(rule_frames.max_inp[:, 0], rule_frames.max_inp[:, 1], "ro")
plt.savefig("plots/burgers2D_rules.png")
plt.show()

rl = MakeRule(input_frames, rule_frames)
rl.get_rule()


def logp_rule(ℓ, η, σ):
    """
    Define the rule function
    """

    cov_test = η ** 2 * pm.gp.cov.Matern32(3, ℓ)

    Kn = cov_test(input_frames.test_inp)
    Kn2 = Kn + σ ** 2 * rl.id_mat_
    Kn_inv = pm.math.matrix_inverse(Kn2)
    Ktn_rule1 = cov_test(rl.rula, input_frames.test_inp)
    prod_rule1 = pm.math.dot(Ktn_rule1, Kn_inv)
    Y_pred_rule1 = pm.math.dot(prod_rule1, input_frames.test_con)

    rule_log_lik = 0
    for i in range(rl.size_rule_all):
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.lt(Y_pred_rule1[i], 1), 1, 0
        )

    rule_log_lik_ext = pm.math.minimum(rule_log_lik, pm.math.abs_(rule_log_lik - 0.01))
    rule_ratio = rule_log_lik_ext / rl.size_rule_all

    return pm.Beta.dist(alpha=1.0, beta=1000.0).logp(rule_ratio)


use_rule = True

with pm.Model() as model:
    ℓ = pm.Gamma("ℓ", alpha=1, beta=1, shape=3)
    η = pm.HalfCauchy("η", beta=1)
    σ = pm.HalfCauchy("σ", beta=1)

    cov = η ** 2 * pm.gp.cov.Matern32(3, ℓ)

    gp = pm.gp.Marginal(cov_func=cov)
    y_ = gp.marginal_likelihood(
        "y_obs", X=input_frames.test_inp, y=input_frames.test_con, noise=σ
    )

    if use_rule:
        LL_rule = pm.Potential("LL_rule", logp_rule(ℓ, η, σ))

with model:
    mp = pm.find_MAP(method="Nelder-Mead")

frames_predict = np.array([0, 21, 43, 64, 85, 106, 128, 149])
predict_frames = FramePic(x, y, times, frames_predict)
predict_frames.inputdata()

mu, _ = gp.predict(predict_frames.test_inp, point=mp, diag=True)

result = xr.DataArray(
    mu.reshape(-1, predict_frames.x.size, predict_frames.y.size),
    dims=["time", "x", "y"],
    coords={"time": frames_predict, "x": x, "y": y},
)

result.plot(col="time", robust=True, aspect=1)
plt.savefig("plots/burgers2D_posterior" + "_rule" * use_rule + "_raw.png")

result1 = result.where(result >= 0, other=0)
result1.plot(col="time", robust=True, aspect=1)
plt.savefig("plots/burgers2D_posterior" + "_rule" * use_rule + "_filtered.png")

