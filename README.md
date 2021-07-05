# Rule-based-Bayesian-regr
Code associated with the paper "Rule-based Bayesian regression"

# One-dimensional analysis

## Installation

Pre-requisites:

1. `git`

2. `conda` python 3.7 environment created and activated using

   ```sh
   conda create -n rules python=3.7
   conda activate rules
   ```

Install dependencies:

```shell
git clone https://github.com/themisbo/Rule-based-Bayesian-regr.git
cd rule-based-bayesian-regression
pip install -r requirements.txt
```

## Getting started

Run the one-dimensional analysis:

```shell
cd rule-based-bayesian-regression/advection-1d
./run_sampling.py
```

Run the two-dimensional analysis:

```shell
cd rule-based-bayesian-regression/burgers-2d
./run_sampling.py
```

## Anticipated run time
The scripts were run with a MacBook Pro (2008): 2.9 GHz 6-Core Intel Core i9, 32 GB 2400 MHz DDR4
```shell
advection-1d without rules ~ 5 minutes
advection-1d with rules ~ 15 minutes
burgers-2d without rules ~ 11 minutes
burgers-2d with rules ~ 11 minutes
```

## References

[1] Bar-Sinai, Y., Hoyer, S., Hickey, J., & Brenner, M. P. (2019). Learning data-driven discretizations for partial differential equations. _Proceedings of the National Academy of Sciences_, _116_(31), 15344-15349.


## Citation
If you use this work or build up on it, please cite https://arxiv.org/abs/2008.00422.
