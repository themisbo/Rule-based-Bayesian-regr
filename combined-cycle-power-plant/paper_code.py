import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pymc3 as pm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_excel("Folds5x2_pp.xlsx")
# df.head()
# df.describe()
# df.corr()["PE"].sort_values()
# df.plot.scatter("AT", "PE")

# Temperature AT
# Pressure AP
# Humidity RH
# Vacuum V
# Electrical energy PE


df_test = df[df["AT"] >= 30]
# df_test.plot.scatter("AH", "CO")

fig, ax = plt.subplots()
ax.scatter(df["AT"], df["PE"], label="Unobserved Data")
ax.scatter(df_test["AT"], df_test["PE"], c="red", label="Observed Data")
ax.set_xlabel("AT")
ax.set_ylabel("PE")
ax.set_title("Data")
ax.legend(ncol=2, fontsize=10)
fig.savefig("plots/CCPP_data.png", dpi=300, bbox_inches="tight", facecolor="w")
fig.show()


# Train-test split
# labels = df[["PE"]]
X = df_test.drop(columns=["PE"])
y = df_test[["PE"]]

scalerX = StandardScaler().fit(X)
scalery = StandardScaler().fit(y)
X = scalerX.transform(X)
y = scalery.transform(y)

X_all = pd.DataFrame(X, columns=df.columns[:4])
y_all = pd.DataFrame(y, columns=df.columns[4:])

X_train = X_all
y_train = y_all

X_orig = scalerX.transform(df.drop(columns=["PE"]))
y_orig = scalery.transform(df[["PE"]])

X_orig = pd.DataFrame(X_orig, columns=df.columns[:4])
y_orig = pd.DataFrame(y_orig, columns=df.columns[4:])


###########################################################
# Simple Bayesian linear regression - all features
###########################################################

AT_val = np.array(X_train["AT"])
V_val = np.array(X_train["V"])
AP_val = np.array(X_train["AP"])
RH_val = np.array(X_train["RH"])
PE_val = np.array(y_train["PE"])

# Define the model
with pm.Model() as model:
    # define priors
    AT_co = pm.Normal("AT_co", 0.0, 10.0, testval=-1.95)
    V_co = pm.Normal("V_co", 0.0, 10.0, testval=-0.24)
    AP_co = pm.Normal("AP_co", 0.0, 10.0, testval=0.17)
    RH_co = pm.Normal("RH_co", 0.0, 10.0, testval=-0.15)
    b = pm.Normal("intercept", 0, 16, testval=340)
    s = pm.Exponential("error", 1, testval=4.5)

    res = AT_co * AT_val + V_co * V_val + AP_co * AP_val + RH_co * RH_val + b

    # predictions
    obs = pm.Normal("observation", res, s, observed=PE_val,)

# Run the MCMC
with model:
    step = pm.Metropolis(scaling=0.01)
    trace = pm.sample(draws=100000, step=step, tune=15000, cores=1, chains=1)

# Get posterior samples
RANDOM_SEED = 10

with model:
    ppc = pm.sample_posterior_predictive(
        trace,
        samples=1000,
        var_names=["intercept", "AT_co", "V_co", "AP_co", "RH_co", "observation",],
        random_seed=RANDOM_SEED,
    )

mu_pp = (ppc["intercept"] + ppc["AT_co"] * AT_val[:, None]).T

mu_pp_mean = (
    ppc["intercept"].mean() + ppc["AT_co"].mean() * X_orig["AT"].to_numpy()[:, None]
).T.reshape(-1)

mu_pp_mean_new = mu_pp_mean * np.sqrt(scalery.var_[0]) + scalery.mean_[0]

fig, ax = plt.subplots()

ax.plot(
    df["AT"].to_numpy(),
    df["PE"].to_numpy(),
    "o",
    ms=4,
    alpha=0.4,
    label="Unobserved Data",
)
ax.plot(
    df_test["AT"].to_numpy(),
    df_test["PE"].to_numpy(),
    "o",
    ms=4,
    alpha=0.4,
    label="Observed Data",
)
for j in range(mu_pp.shape[0]):
    mu_pp_sample = (
        ppc["intercept"][j] + ppc["AT_co"][j] * X_orig["AT"].to_numpy()[:, None]
    ).T.reshape(-1)
    mu_pp_sample_new = mu_pp_sample * np.sqrt(scalery.var_[0]) + scalery.mean_[0]
    ax.plot(df["AT"].to_numpy(), mu_pp_sample_new, alpha=0.3, c="g")
ax.plot(df["AT"].to_numpy(), mu_pp_mean_new, label="Mean outcome", alpha=0.6, c="r")
ax.set_xlabel("Predictor (AT)")
ax.set_ylabel("Outcome (PE)")
ax.set_title("Posterior samples")
ax.legend(ncol=2, fontsize=10)
fig.savefig("plots/CCPP_norules.png", dpi=300, bbox_inches="tight", facecolor="w")

# Metrics
pred = (
    trace["AT_co"].mean() * X_orig["AT"].to_numpy()
    + trace["V_co"].mean() * X_orig["V"].to_numpy()
    + trace["AP_co"].mean() * X_orig["AP"].to_numpy()
    + trace["RH_co"].mean() * X_orig["RH"].to_numpy()
    + trace["intercept"].mean()
)

rmse = mean_squared_error(y_orig["PE"].to_numpy(), pred, squared=False)
mse = mean_squared_error(y_orig["PE"].to_numpy(), pred)
mae = mean_absolute_error(y_orig["PE"].to_numpy(), pred)

with open("metrics_paper.txt", "a") as text_file:
    print("Metrics without rules:", file=text_file)
    print(f"rmse: {rmse}", file=text_file)
    print(f"mse: {mse}", file=text_file)
    print(f"mae: {mae}", file=text_file)

###########################################################
# Rule-based Bayesian linear regression
###########################################################

# Rule discretization
no_points = 25
xlow = X_all.describe()["AT"]["min"]
xmid1 = X_all.describe()["AT"]["25%"]
xmid = X_all.describe()["AT"]["50%"]
xmid2 = X_all.describe()["AT"]["75%"]
xhi = X_all.describe()["AT"]["max"]

rule_first = np.linspace(xlow, xmid1, no_points)
rule_second = np.linspace(xmid2, xhi, no_points)

ymid_min = y_all.describe()["PE"]["25%"]
ymid = y_all.describe()["PE"]["50%"]
ymid_max = y_all.describe()["PE"]["75%"]


def logp_rule_nopm(
    AT_co, b_lat,
):

    points_r1 = rule_first * AT_co + b_lat
    points_r2 = rule_second * AT_co + b_lat

    rule_log_lik = 0
    for i in range(no_points):
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.lt(points_r1[i], ymid), 1, 0
        )
        rule_log_lik = rule_log_lik + pm.math.switch(
            pm.math.gt(points_r2[i], ymid_min), 1, 0
        )

    rule_ratio = rule_log_lik / (2 * no_points)

    return pm.Beta.dist(alpha=1.0, beta=10000.0).logp(rule_ratio)


with pm.Model() as model:
    # define priors
    AT_co = pm.Normal("AT_co", 0.0, 10.0, testval=-2.0)
    V_co = pm.Normal("V_co", 0.0, 10.0)
    AP_co = pm.Normal("AP_co", 0.0, 10.0)
    RH_co = pm.Normal("RH_co", 0.0, 10.0)
    b = pm.Normal("intercept", 0, 16.0)
    s = pm.Exponential("error", 1)

    b_lat = pm.Normal("intercept_lat", 0, 16.0, testval=0.0)

    res = AT_co * AT_val + b

    # predictions
    obs = pm.Normal("observation", res, s, observed=PE_val,)
    LL_rule = pm.Potential("LL_rule", logp_rule_nopm(AT_co, b_lat))

with model:
    step = pm.Metropolis(scaling=0.01)
    trace = pm.sample(draws=100000, step=step, tune=30000, cores=1, chains=1)

# Get posterior samples
RANDOM_SEED = 10

with model:
    ppc = pm.sample_posterior_predictive(
        trace,
        samples=1000,
        var_names=["intercept", "AT_co", "observation",],
        random_seed=RANDOM_SEED,
    )
mu_pp = (ppc["intercept"] + ppc["AT_co"] * AT_val[:, None]).T
mu_pp_mean = (
    ppc["intercept"].mean() + ppc["AT_co"].mean() * X_orig["AT"].to_numpy()[:, None]
).T.reshape(-1)

mu_pp_mean_new = mu_pp_mean * np.sqrt(scalery.var_[0]) + scalery.mean_[0]

fig, ax = plt.subplots()

ax.plot(
    df["AT"].to_numpy(),
    df["PE"].to_numpy(),
    "o",
    ms=4,
    alpha=0.4,
    label="Unobserved Data",
)
ax.plot(
    df_test["AT"].to_numpy(),
    df_test["PE"].to_numpy(),
    "o",
    ms=4,
    alpha=0.4,
    label="Observed Data",
)
for j in range(mu_pp.shape[0]):
    mu_pp_sample = (
        ppc["intercept"][j] + ppc["AT_co"][j] * X_orig["AT"].to_numpy()[:, None]
    ).T.reshape(-1)
    mu_pp_sample_new = mu_pp_sample * np.sqrt(scalery.var_[0]) + scalery.mean_[0]
    ax.plot(df["AT"].to_numpy(), mu_pp_sample_new, alpha=0.3, c="g")
ax.plot(df["AT"].to_numpy(), mu_pp_mean_new, label="Mean outcome", alpha=0.6, c="r")
ax.set_xlabel("Predictor (AT)")
ax.set_ylabel("Outcome (PE)")
ax.set_title("Posterior samples")
ax.legend(ncol=2, fontsize=10)
xmid1_new = xmid1 * np.sqrt(scalerX.var_[0]) + scalerX.mean_[0]
xmid2_new = xmid2 * np.sqrt(scalerX.var_[0]) + scalerX.mean_[0]
ymid_min_new = ymid_min * np.sqrt(scalery.var_[0]) + scalery.mean_[0]
ymid_new = ymid * np.sqrt(scalery.var_[0]) + scalery.mean_[0]
plt.axvline(xmid1_new, c="red", linewidth=2)
plt.axvline(xmid2_new, c="red", linewidth=2)
plt.axhline(ymid_min_new, c="red", linewidth=2)
plt.axhline(ymid_new, c="red", linewidth=2)
fig.savefig("plots/CCPP_rules1.png", dpi=300, bbox_inches="tight", facecolor="w")


# Metrics
pred = (
    trace["AT_co"].mean() * X_orig["AT"].to_numpy()
    + trace["V_co"].mean() * X_orig["V"].to_numpy()
    + trace["AP_co"].mean() * X_orig["AP"].to_numpy()
    + trace["RH_co"].mean() * X_orig["RH"].to_numpy()
    + trace["intercept"].mean()
)

rmse = mean_squared_error(y_orig["PE"].to_numpy(), pred, squared=False)
mse = mean_squared_error(y_orig["PE"].to_numpy(), pred)
mae = mean_absolute_error(y_orig["PE"].to_numpy(), pred)


with open("metrics_paper.txt", "a") as text_file:
    print("Metrics with rules:", file=text_file)
    print(f"rmse: {rmse}", file=text_file)
    print(f"mse: {mse}", file=text_file)
    print(f"mae: {mae}", file=text_file)

pooled_loo = az.loo(trace, model)
pooled_loo.loo
