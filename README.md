# Strucutral uncertainty for reaction network inference

**A preprint of this work is available on [arXiv](https://arxiv.org/abs/2505.15653).**

Project during MATRIX program "Parameter Identifiability in Mathematical Biology" (Sep 2024).

This repository provides Julia code for inferring the structure of a chemical reaction network (CRN) from noisy time series data of species concentrations. Our approach is based on constructing a system of ordinary differential equations (ODEs) from a library of candidate reactions, and subsequently performing parameter inference to estimate the reaction rate constants. Reactions with non-negligible rate constant estimates are deduced to be present in the network. 

The Jupyter notebook `crn_inference_tutorial.ipynb` features a standalone, simplified demo of CRN inference. The `src/` directory provides functionality for CRN data simulation, parameter inference, network inference, and hyperparameter tuning. See `test/` for applications.

Parameter estimation is done in a frequentist fashion via maximising a penalised likelihood. To this end, we use a multi-start first-order optimisation algorithm, aided by automatic differentiation. A penalty function is used to encourage parsimonious model fits to the data. The implemented penalty functions are as follows:
- $L_1$ penalty on the original scale, i.e. an exponential prior,
- $L_1$ penalty on a shifted-log scale, i.e. a shifted-exponential prior on the log scale,
- approximate $L_0$ penalty, which is approximately proportional to the effective number of reactions *a la* Bayesian information criterion, 
- [horseshoe-like prior](https://arxiv.org/abs/1702.07400), which is a closed-form approximation of the horseshoe prior commonly used as a sparse prior in Bayesian hierarchical modelling.

Our approach shares some similarities to that of [Hoffmann et al. (2019)](https://doi.org/10.1063/1.5066099), who use a $L_1$ penalty (original scale). One notable difference is that Hoffmann et al. avoid numerically solving the ODE, and instead design a loss function based on derivative error, which relies on estimating derivatives from time series data. We also note that for our case study, the $L_1$ penalty (original scale) had the poorest overall performance out of the penalty functions considered.

Note: These descriptions are outdated.

Deciding whether a reaction rate constant estimate is negligible or not is often done subjectively, e.g. in Hoffmann et al. (2019), a cutoff is manually chosen by visual inspection. We propose an automatic procedure for cutoff determination, which we implement in `src/inference.jl`. For our simulation study, we find that this automated cutoff determination performs sufficiently well for all penalty functions except the $L_1$ penalty; see `test/vary_kvals/README.md`.

Some methods details are given in `details.md`.
