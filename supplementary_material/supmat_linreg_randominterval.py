import matplotlib.pyplot as plt

plt.style.use("ggplot")
import numpy as np
import pymc3 as pm

### Synthetic data generation ###
sample_size = 500
sigma_e = 3.0  # true value of parameter error sigma
np.random.seed(1)
random_num_generator = np.random.RandomState(0)

x_all = 10.0 * random_num_generator.rand(sample_size)
e = random_num_generator.normal(0, sigma_e, len(x_all))
y_all = 1.0 + 2.0 * x_all + e  # a = 1.0; b = 2.0; y = a + b*x

xor = list(enumerate(x_all))

inds = np.random.choice(len(xor), 50, replace=False)

x = x_all[inds]
y = y_all[inds]

x_disc = np.linspace(0, 10, 20)
y_true = 1.0 + 2.0 * x_disc

fig = plt.figure(figsize=(6, 4))
ax = fig.gca()
plt.scatter(x_all, y_all, color="mediumaquamarine", label="Non visible data")
plt.scatter(x, y, color="blue", label="Visible data")
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


#############################################
### Rule-based Bayesian linear regression ###
#############################################


def logp_rule(a, b):

    rule_log_lik = 0
    rule_log_lik = rule_log_lik + pm.math.switch(
        pm.math.or_(
            pm.math.or_(pm.math.lt(a + b * 0, 0), pm.math.lt(a + b * 1, 0)),
            pm.math.or_(pm.math.gt(a + b * 0, 4), pm.math.gt(a + b * 1, 4)),
        ),
        1,
        0,
    )

    rule_log_lik = rule_log_lik + pm.math.switch(
        pm.math.or_(
            pm.math.or_(pm.math.lt(a + b * 9, 18), pm.math.lt(a + b * 10, 18)),
            pm.math.or_(pm.math.gt(a + b * 9, 22), pm.math.gt(a + b * 10, 22)),
        ),
        1,
        0,
    )

    rule_ratio = rule_log_lik / 2

    return pm.Beta.dist(alpha=1.0, beta=5.0).logp(rule_ratio)


with pm.Model() as rule_model:

    # Priors for unknown model parameters
    a = pm.Normal("alpha", mu=0.5, sigma=0.5)
    b = pm.Normal("beta", mu=0.5, sigma=0.5)

    # Expected value of outcome
    mu = a + b * x

    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma_e, observed=y)
    LL_rule = pm.Potential("LL_rule", logp_rule(a, b))

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

# Note: The rule-based variation performed marginally better than the standard Bayesian
# counterpart, though the difference was not as severe as with the case that
# data were not available for a significant portion of the left- and right-hand side of the plot.
