import matplotlib.pyplot as plt

plt.style.use("ggplot")
import numpy as np
import pymc3 as pm

### Synthetic data generation ###
sample_size = 500
sigma_e = 3.0  # true value of parameter error sigma
np.random.seed(1)
random_num_generator = np.random.RandomState(0)
x = 10.0 * random_num_generator.rand(sample_size)
x_lt = x[x < 4]
x_gt = x[x > 5]
x = x[x > 4]
x = x[x < 5]
e = random_num_generator.normal(0, sigma_e, len(x))
y = 1.0 + 2.0 * x + e  # a = 1.0; b = 2.0; y = a + b*x

y_lt = 1.0 + 2.0 * x_lt + random_num_generator.normal(0, sigma_e, len(x_lt))
y_gt = 1.0 + 2.0 * x_gt + random_num_generator.normal(0, sigma_e, len(x_gt))

x_disc = np.linspace(0, 10, 20)
y_true = 1.0 + 2.0 * x_disc

fig = plt.figure(figsize=(6, 4))
ax = fig.gca()
plt.scatter(x, y, color="blue", label="Visible data")
plt.scatter(x_lt, y_lt, color="mediumaquamarine", label="Non visible data")
plt.scatter(x_gt, y_gt, color="mediumaquamarine")
plt.plot(
    x_disc, y_true, linewidth=3, color="darkolivegreen", label="True regression line"
)
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.legend()

###########################################
### Standard Bayesian linear regression ###
###########################################

with pm.Model() as basic_model:

    # Priors for unknown model parameters
    a = pm.Normal("alpha", mu=0.5, sigma=0.5)
    b = pm.Normal("beta", mu=0.5, sigma=0.5)

    # Expected value of outcome
    mu = a + b * x

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma_e, observed=y)


# MCMC
with basic_model:
    step = pm.Metropolis()
    trace = pm.sample(draws=100000, tune=20000, step=step, cores=1, chains=1)

# Vizualization
thin = 100
xvals = np.linspace(0, 10, 20)

fig = plt.figure(figsize=(6, 4))
ax = fig.gca()
for iter in range(int(trace["alpha"].shape[0] / thin)):
    # select alternate samples to decrease auto corr for now
    a = trace["alpha"][iter * thin]
    b = trace["beta"][iter * thin]
    yvals = a + b * xvals
    plt.plot(xvals, yvals, color="red", alpha=0.1)
plt.plot(xvals, yvals, color="red", alpha=1, label="Posterior regression lines")
ymean = trace["alpha"].mean() + trace["beta"].mean() * xvals
plt.plot(xvals, ymean, color="yellow", label="Mean posterior regression line")
plt.scatter(x, y, color="blue", label="Data")
plt.plot(
    x_disc, y_true, linewidth=3, color="darkolivegreen", label="True regression line"
)
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.legend()
plt.show()

####################################################################
### Rule-based Bayesian linear regression - rule hyperparameters ###
####################################################################


def logp_rule(a, b, xlow, xhi, ylow, yhi):

    rule_log_lik = 0
    rule_log_lik = rule_log_lik + pm.math.switch(
        pm.math.or_(
            pm.math.or_(pm.math.lt(a + b * 0, 0), pm.math.lt(a + b * xlow, 0)),
            pm.math.or_(pm.math.gt(a + b * 0, ylow), pm.math.gt(a + b * xlow, ylow)),
        ),
        1,
        0,
    )

    rule_log_lik = rule_log_lik + pm.math.switch(
        pm.math.or_(
            pm.math.or_(pm.math.lt(a + b * xhi, yhi), pm.math.lt(a + b * 10, yhi)),
            pm.math.or_(pm.math.gt(a + b * xhi, 22), pm.math.gt(a + b * 10, 22)),
        ),
        1,
        0,
    )

    rule_ratio = rule_log_lik / 2

    return pm.Beta.dist(alpha=1.0, beta=100.0).logp(rule_ratio)


with pm.Model() as rule_model:

    # Priors for unknown model parameters
    a = pm.Normal("alpha", mu=0.5, sigma=0.5)
    b = pm.Normal("beta", mu=0.5, sigma=0.5)
    xlow = pm.Normal("xlow", mu=1.5, sigma=0.5)
    xhi = pm.Normal("xhi", mu=8.5, sigma=0.5)
    ylow = pm.Normal("ylow", mu=4.5, sigma=0.5)
    yhi = pm.Normal("yhi", mu=18.5, sigma=0.5)
    # sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu = a + b * x

    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma_e, observed=y)
    LL_rule = pm.Potential("LL_rule", logp_rule(a, b, xlow, xhi, ylow, yhi))

# MCMC
with rule_model:
    step = pm.Metropolis()
    trace = pm.sample(draws=100000, tune=20000, step=step, cores=1, chains=1)


thin = 100

fig = plt.figure(figsize=(6, 4))
ax = fig.gca()
for iter in range(int(trace["alpha"].shape[0] / thin)):
    # select alternate samples to decrease auto corr for now
    a = trace["alpha"][iter * thin]
    b = trace["beta"][iter * thin]
    yvals = a + b * xvals
    plt.plot(xvals, yvals, color="red", alpha=0.1)
plt.plot(xvals, yvals, color="red", alpha=1, label="Posterior regression lines")
ymean = trace["alpha"].mean() + trace["beta"].mean() * xvals
plt.plot(xvals, ymean, color="yellow", label="Mean posterior regression line")
plt.scatter(x, y, color="blue", label="Data")
plt.plot(
    x_disc, y_true, linewidth=3, color="darkolivegreen", label="True regression line"
)
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.legend()
plt.show()


####################################################################
### Rule-based Bayesian linear regression - rule hyperparameters ###
######################### Truncated Normal priors ##################
####################################################################


def logp_rule(a, b, xlow, xhi, ylow, yhi):

    rule_log_lik = 0
    rule_log_lik = rule_log_lik + pm.math.switch(
        pm.math.or_(
            pm.math.or_(pm.math.lt(a + b * 0, 0), pm.math.lt(a + b * xlow, 0)),
            pm.math.or_(pm.math.gt(a + b * 0, ylow), pm.math.gt(a + b * xlow, ylow)),
        ),
        1,
        0,
    )

    rule_log_lik = rule_log_lik + pm.math.switch(
        pm.math.or_(
            pm.math.or_(pm.math.lt(a + b * xhi, yhi), pm.math.lt(a + b * 10, yhi)),
            pm.math.or_(pm.math.gt(a + b * xhi, 22), pm.math.gt(a + b * 10, 22)),
        ),
        1,
        0,
    )

    rule_ratio = rule_log_lik / 2

    return pm.Beta.dist(alpha=1.0, beta=100.0).logp(rule_ratio)


with pm.Model() as rule_model:

    # Priors for unknown model parameters
    a = pm.Normal("alpha", mu=0.5, sigma=0.5)
    b = pm.Normal("beta", mu=0.5, sigma=0.5)
    xlow = pm.TruncatedNormal("xlow", mu=1.5, sigma=0.5, lower=0)
    xhi = pm.TruncatedNormal("xhi", mu=8.5, sigma=0.5, upper=10)
    ylow = pm.Normal("ylow", mu=4.5, sigma=0.5)
    yhi = pm.Normal("yhi", mu=18.5, sigma=0.5)
    # sigma = pm.HalfNormal('sigma', sigma=1)

    # Expected value of outcome
    mu = a + b * x

    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma_e, observed=y)
    LL_rule = pm.Potential("LL_rule", logp_rule(a, b, xlow, xhi, ylow, yhi))

# MCMC
with rule_model:
    step = pm.Metropolis()
    trace = pm.sample(draws=100000, tune=20000, step=step, cores=1, chains=1)


thin = 100

fig = plt.figure(figsize=(6, 4))
ax = fig.gca()
for iter in range(int(trace["alpha"].shape[0] / thin)):
    # select alternate samples to decrease auto corr for now
    a = trace["alpha"][iter * thin]
    b = trace["beta"][iter * thin]
    yvals = a + b * xvals
    plt.plot(xvals, yvals, color="red", alpha=0.1)
plt.plot(xvals, yvals, color="red", alpha=1, label="Posterior regression lines")
ymean = trace["alpha"].mean() + trace["beta"].mean() * xvals
plt.plot(xvals, ymean, color="yellow", label="Mean posterior regression line")
plt.scatter(x, y, color="blue", label="Data")
plt.plot(
    x_disc, y_true, linewidth=3, color="darkolivegreen", label="True regression line"
)
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.legend()
plt.show()

# Note: There was no significant difference by using the truncated Normal priors in the result.
# The posterior shape is virtually identical with the Guassian priors.
