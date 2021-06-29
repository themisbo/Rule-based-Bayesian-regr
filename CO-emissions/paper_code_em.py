import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import arviz as az
import pymc3 as pm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load the data
df = pd.read_csv("gt_2013.csv", index_col=None, header=0)

# df.head()
# df.describe()
# df.corr()["CO"].sort_values()
# df.plot.scatter("AH", "CO")

df_test = df[df["AH"] >= 97]

# df_test.plot.scatter("AH", "CO")

fig, ax = plt.subplots()
ax.scatter(df["AH"], df["CO"], label="Unobserved Data")
ax.scatter(df_test["AH"], df_test["CO"], c="red", label="Observed Data")
ax.set_xlabel("AH")
ax.set_ylabel("CO")
ax.set_title("Data")
ax.legend(ncol=2, fontsize=10)
fig.savefig("plots/gas_em_data.png", dpi=300, bbox_inches="tight", facecolor="w")
fig.show()

labels = df_test[["CO", "NOX"]]

X = df_test.drop(columns=["CO", "NOX"])
y = df_test[["CO", "NOX"]]

scalerX = StandardScaler().fit(X)
scalery = StandardScaler().fit(y)
X = scalerX.transform(X)
y = scalery.transform(y)

X_all = pd.DataFrame(X, columns=df.columns[:9])
y_all = pd.DataFrame(y, columns=df.columns[9:])

X_train = X_all
y_train = y_all

X_orig = scalerX.transform(df.drop(columns=["CO", "NOX"]))
y_orig = scalery.transform(df[["CO", "NOX"]])

X_orig = pd.DataFrame(X_orig, columns=df.columns[:9])
y_orig = pd.DataFrame(y_orig, columns=df.columns[9:])

###########################################################
# Simple Bayesian linear regression
###########################################################

AH_val = np.array(X_train["AH"])
AT_val = np.array(X_train["AT"])
AFDP_val = np.array(X_train["AFDP"])
GTEP_val = np.array(X_train["GTEP"])
CO_val = np.array(y_train["CO"])
NOX_val = np.array(y_train["NOX"])

with pm.Model() as model:
    AT_co = pm.Normal("AT_co", 0.0, 10.0, testval=-0.03)
    AH_co = pm.Normal("AH_co", 0.0, 10.0, testval=0.01)
    AFDP_co = pm.Normal("AFDP_co", 0.0, 10.0, testval=0.1)
    GTEP_co = pm.Normal("GTEP_co", 0.0, 10.0, testval=-0.35)
    b = pm.Normal("intercept", 0, 16, testval=11)
    s = pm.Exponential("error", 1, testval=2.1)

    res = AT_co * AT_val + AH_co * AH_val + AFDP_co * AFDP_val + GTEP_co * GTEP_val + b

    obs = pm.Normal("observation", res, s, observed=CO_val,)

with model:
    step = pm.Metropolis(scaling=0.01)
    trace = pm.sample(draws=100000, step=step, tune=15000, cores=1, chains=1)

RANDOM_SEED = 10

with model:
    ppc = pm.sample_posterior_predictive(
        trace,
        samples=1000,
        var_names=["intercept", "AT_co", "AH_co", "AFDP_co", "GTEP_co", "observation"],
        random_seed=RANDOM_SEED,
    )

mu_pp = (ppc["intercept"] + ppc["AH_co"] * AH_val[:, None]).T

mu_pp_mean = (
    ppc["intercept"].mean() + ppc["AH_co"].mean() * X_orig["AH"].to_numpy()[:, None]
).T.reshape(-1)

mu_pp_mean_new = mu_pp_mean * np.sqrt(scalery.var_[0]) + scalery.mean_[0]

fig, ax = plt.subplots()
ax.plot(
    df["AH"].to_numpy(), df["CO"].to_numpy(), "o", ms=4, alpha=0.4, label="Unseen Data",
)
ax.plot(
    df_test["AH"].to_numpy(),
    df_test["CO"].to_numpy(),
    "o",
    ms=4,
    alpha=0.4,
    label="Observed Data",
)
for j in range(mu_pp.shape[0]):
    mu_pp_sample = (
        ppc["intercept"][j] + ppc["AH_co"][j] * X_orig["AH"].to_numpy()[:, None]
    ).T.reshape(-1)
    mu_pp_sample_new = mu_pp_sample * np.sqrt(scalery.var_[0]) + scalery.mean_[0]
    ax.plot(df["AH"].to_numpy(), mu_pp_sample_new, alpha=0.3, c="g")
ax.plot(df["AH"].to_numpy(), mu_pp_mean_new, label="Mean outcome", alpha=0.6, c="r")
ax.set_xlabel("Predictor (AH)")
ax.set_ylabel("Outcome (CO)")
ax.set_title("Posterior samples")
ax.legend(ncol=2, fontsize=10)
fig.savefig(
    "plots/gas_em_post_norules.png", dpi=300, bbox_inches="tight", facecolor="w"
)

# Metrics
pred = (
    trace["AT_co"].mean() * X_orig["AT"].to_numpy()
    + trace["AH_co"].mean() * X_orig["AH"].to_numpy()
    + trace["AFDP_co"].mean() * X_orig["AFDP"].to_numpy()
    + trace["GTEP_co"].mean() * X_orig["GTEP"].to_numpy()
    + trace["intercept"].mean()
)

rmse = mean_squared_error(y_orig["CO"].to_numpy(), pred, squared=False)
mse = mean_squared_error(y_orig["CO"].to_numpy(), pred)
mae = mean_absolute_error(y_orig["CO"].to_numpy(), pred)

with open("metrics_paper.txt", "a") as text_file:
    print("Metrics without rules:", file=text_file)
    print(f"rmse: {rmse}", file=text_file)
    print(f"mse: {mse}", file=text_file)
    print(f"mae: {mae}", file=text_file)


################################
##### AH rule ##############
################################

no_points = 25

xlow = X_all.describe()["AH"]["min"]
xmid1 = X_all.describe()["AH"]["25%"]
xmid = X_all.describe()["AH"]["50%"]
xmid2 = X_all.describe()["AH"]["75%"]
xhi = X_all.describe()["AH"]["max"]

rule_first = np.linspace(xlow, xmid1, no_points)
rule_second = np.linspace(xmid2, xhi, no_points)


ymid_low = y_all.describe()["CO"]["25%"]
ymid = y_all.describe()["CO"]["50%"]
ymid_high = y_all.describe()["CO"]["75%"]


def logp_rule_nopm(
    AH_co, b_lat,
):

    points_r1 = rule_first * AH_co + b_lat
    points_r2 = rule_second * AH_co + b_lat

    rule_log_lik = 0
    for i in range(no_points):
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.gt(points_r1[i], ymid), 1, 0
        )
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.lt(points_r2[i], ymid), 1, 0
        )

    rule_ratio = rule_log_lik / (2 * no_points)

    return pm.Beta.dist(alpha=1.0, beta=1000.0).logp(rule_ratio)


with pm.Model() as model:
    # define priors
    AT_co = pm.Normal("AT_co", 0.0, 10.0)
    AH_co = pm.Normal("AH_co", 0.0, 10.0, testval=0.035)
    AFDP_co = pm.Normal("AFDP_co", 0.0, 10.0)
    GTEP_co = pm.Normal("GTEP_co", 0.0, 10.0)
    b = pm.Normal("intercept", 0, 16, testval=-0.4)
    s = pm.Exponential("error", 1, testval=2.5)

    b_lat = pm.Normal("intercept_lat", 0, 16, testval=-0.4)

    res = AT_co * AT_val + AH_co * AH_val + AFDP_co * AFDP_val + GTEP_co * GTEP_val + b

    obs = pm.Normal("observation", res, s, observed=CO_val,)
    LL_rule = pm.Potential("LL_rule", logp_rule_nopm(AH_co, b_lat))

with model:
    step = pm.Metropolis(scaling=0.01)
    trace = pm.sample(draws=100000, step=step, tune=30000, cores=1, chains=1)


RANDOM_SEED = 10

with model:
    ppc = pm.sample_posterior_predictive(
        trace,
        samples=1000,
        var_names=["intercept", "AT_co", "AH_co", "observation"],
        random_seed=RANDOM_SEED,
    )

mu_pp = (ppc["intercept"] + ppc["AH_co"] * AH_val[:, None]).T

mu_pp_mean = (
    ppc["intercept"].mean() + ppc["AH_co"].mean() * X_orig["AH"].to_numpy()[:, None]
).T.reshape(-1)

mu_pp_mean_new = mu_pp_mean * np.sqrt(scalery.var_[0]) + scalery.mean_[0]

fig, ax = plt.subplots()

ax.plot(df["AH"].to_numpy(), df["CO"].to_numpy(), "o", ms=4, alpha=0.4, label="Data")
ax.plot(
    df_test["AH"].to_numpy(),
    df_test["CO"].to_numpy(),
    "o",
    ms=4,
    alpha=0.4,
    label="Observed Data",
)
for j in range(mu_pp.shape[0]):
    mu_pp_sample = (
        trace["intercept"][::100][j]
        + trace["AH_co"][::100][j] * X_orig["AH"].to_numpy()[:, None]
    ).T.reshape(-1)
    mu_pp_sample_new = mu_pp_sample * np.sqrt(scalery.var_[0]) + scalery.mean_[0]
    ax.plot(df["AH"].to_numpy(), mu_pp_sample_new, alpha=0.3, c="g")
ax.plot(df["AH"].to_numpy(), mu_pp_mean_new, label="Mean outcome", alpha=0.6, c="r")
ax.set_xlabel("Predictor (AH)")
ax.set_ylabel("Outcome (CO)")
ax.set_title("Posterior predictive checks")
ax.legend(ncol=2, fontsize=10)
xmid1_new = xmid1 * np.sqrt(scalerX.var_[2]) + scalerX.mean_[2]
xmid2_new = xmid2 * np.sqrt(scalerX.var_[2]) + scalerX.mean_[2]
ymid_new = ymid * np.sqrt(scalery.var_[0]) + scalery.mean_[0]
plt.axvline(xmid1_new, c="red", linewidth=2)
plt.axvline(xmid2_new, c="red", linewidth=2)
plt.axhline(ymid_new, c="red", linewidth=2)
fig.savefig("plots/gas_em_post_rules1.png", dpi=300, bbox_inches="tight", facecolor="w")
##########

# Metrics
pred = (
    trace["AT_co"].mean() * X_orig["AT"].to_numpy()
    + trace["AH_co"].mean() * X_orig["AH"].to_numpy()
    + trace["AFDP_co"].mean() * X_orig["AFDP"].to_numpy()
    + trace["GTEP_co"].mean() * X_orig["GTEP"].to_numpy()
    + trace["intercept"].mean()
)

rmse = mean_squared_error(y_orig["CO"].to_numpy(), pred, squared=False)
mse = mean_squared_error(y_orig["CO"].to_numpy(), pred)
mae = mean_absolute_error(y_orig["CO"].to_numpy(), pred)

pooled_loo = az.loo(trace, model)

pooled_loo.loo

with open("metrics_paper.txt", "a") as text_file:
    print("Metrics with rules, AH common:", file=text_file)
    print(f"rmse: {rmse}", file=text_file)
    print(f"mse: {mse}", file=text_file)
    print(f"mae: {mae}", file=text_file)


##############################################
##### AH rule and lowest CO ##############
##############################################

no_points = 25

xlow = X_all.describe()["AH"]["min"]
xmid1 = X_all.describe()["AH"]["25%"]
xmid = X_all.describe()["AH"]["50%"]
xmid2 = X_all.describe()["AH"]["75%"]
xhi = X_all.describe()["AH"]["max"]

xlow_orig = X_orig.describe()["AH"]["min"]
xhi_orig = X_orig.describe()["AH"]["max"]

rule_first = np.linspace(xlow, xmid1, no_points)
rule_second = np.linspace(xmid2, xhi, no_points)


ymid_low = y_all.describe()["CO"]["25%"]
ymid = y_all.describe()["CO"]["50%"]
ymid_high = y_all.describe()["CO"]["75%"]

y_lowest = -scalery.mean_[0] / np.sqrt(
    scalery.var_[0]
)  # This is 0 in the standard scale
rule_full = np.linspace(xlow_orig, xhi_orig, no_points)


def logp_rule_nopm(
    AH_co, b_lat,
):

    points_r1 = rule_first * AH_co + b_lat
    points_r2 = rule_second * AH_co + b_lat
    points_full = rule_full * AH_co + b_lat

    rule_log_lik = 0
    for i in range(no_points):
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.gt(points_r1[i], ymid), 1, 0
        )
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.lt(points_r2[i], ymid), 1, 0
        )
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.lt(points_full[i], y_lowest), 1, 0
        )

    rule_ratio = rule_log_lik / (3 * no_points)

    return pm.Beta.dist(alpha=1.0, beta=1000.0).logp(rule_ratio)


with pm.Model() as model:
    # define priors
    AT_co = pm.Normal("AT_co", 0.0, 10.0)
    AH_co = pm.Normal("AH_co", 0.0, 10.0, testval=0.035)
    AFDP_co = pm.Normal("AFDP_co", 0.0, 10.0)
    GTEP_co = pm.Normal("GTEP_co", 0.0, 10.0)
    b = pm.Normal("intercept", 0, 16, testval=-0.4)
    s = pm.Exponential("error", 1, testval=2.5)

    b_lat = pm.Normal("intercept_lat", 0, 16, testval=-0.4)

    res = AT_co * AT_val + AH_co * AH_val + AFDP_co * AFDP_val + GTEP_co * GTEP_val + b

    # predictions
    obs = pm.Normal("observation", res, s, observed=CO_val,)
    LL_rule = pm.Potential("LL_rule", logp_rule_nopm(AH_co, b_lat))

with model:
    step = pm.Metropolis(scaling=0.01)
    trace = pm.sample(draws=100000, step=step, tune=30000, cores=1, chains=1)


RANDOM_SEED = 10

with model:
    ppc = pm.sample_posterior_predictive(
        trace,
        samples=1000,
        var_names=["intercept", "AT_co", "AH_co", "observation"],
        random_seed=RANDOM_SEED,
    )

mu_pp = (ppc["intercept"] + ppc["AH_co"] * AH_val[:, None]).T

mu_pp_mean = (
    ppc["intercept"].mean() + ppc["AH_co"].mean() * X_orig["AH"].to_numpy()[:, None]
).T.reshape(-1)

mu_pp_mean_new = mu_pp_mean * np.sqrt(scalery.var_[0]) + scalery.mean_[0]

fig, ax = plt.subplots()

ax.plot(df["AH"].to_numpy(), df["CO"].to_numpy(), "o", ms=4, alpha=0.4, label="Data")
ax.plot(
    df_test["AH"].to_numpy(),
    df_test["CO"].to_numpy(),
    "o",
    ms=4,
    alpha=0.4,
    label="Observed Data",
)
for j in range(mu_pp.shape[0]):
    mu_pp_sample = (
        trace["intercept"][::100][j]
        + trace["AH_co"][::100][j] * X_orig["AH"].to_numpy()[:, None]
    ).T.reshape(-1)
    mu_pp_sample_new = mu_pp_sample * np.sqrt(scalery.var_[0]) + scalery.mean_[0]
    ax.plot(df["AH"].to_numpy(), mu_pp_sample_new, alpha=0.3, c="g")
ax.plot(df["AH"].to_numpy(), mu_pp_mean_new, label="Mean outcome", alpha=0.6, c="r")
ax.set_xlabel("Predictor (AH)")
ax.set_ylabel("Outcome (CO)")
ax.set_title("Posterior predictive checks")
ax.legend(ncol=2, fontsize=10)
xmid1_new = xmid1 * np.sqrt(scalerX.var_[2]) + scalerX.mean_[2]
xmid2_new = xmid2 * np.sqrt(scalerX.var_[2]) + scalerX.mean_[2]
ymid_new = ymid * np.sqrt(scalery.var_[0]) + scalery.mean_[0]
plt.axvline(xmid1_new, c="red", linewidth=2)
plt.axvline(xmid2_new, c="red", linewidth=2)
plt.axhline(ymid_new, c="red", linewidth=2)
plt.axhline(0, c="red", linewidth=2)
fig.savefig("plots/gas_em_post_rules2.png", dpi=300, bbox_inches="tight", facecolor="w")
##########

# Metrics
pred = (
    trace["AT_co"].mean() * X_orig["AT"].to_numpy()
    + trace["AH_co"].mean() * X_orig["AH"].to_numpy()
    + trace["AFDP_co"].mean() * X_orig["AFDP"].to_numpy()
    + trace["GTEP_co"].mean() * X_orig["GTEP"].to_numpy()
    + trace["intercept"].mean()
)

rmse = mean_squared_error(y_orig["CO"].to_numpy(), pred, squared=False)
mse = mean_squared_error(y_orig["CO"].to_numpy(), pred)
mae = mean_absolute_error(y_orig["CO"].to_numpy(), pred)

with open("metrics_paper.txt", "a") as text_file:
    print("Metrics with rules, AH common, y lowest:", file=text_file)
    print(f"rmse: {rmse}", file=text_file)
    print(f"mse: {mse}", file=text_file)
    print(f"mae: {mae}", file=text_file)
