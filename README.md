# Strucutral uncertainty for reaction network inference

**A preprint of this work is available on [arXiv](https://arxiv.org/abs/2505.15653).**

This is a continuation of a project originating from the MATRIX program "Parameter Identifiability in Mathematical Biology" (Sep 2024).

This repository provides Julia code for inferring the structure of a chemical reaction network (CRN) from noisy time series data of species concentrations. Our approach is based on constructing a system of ordinary differential equations (ODEs) from a library of candidate reactions, and subsequently estimate reaction rate constants with penalised optimisation. Reactions with non-negligible rate constant estimates are deduced to be present in the CRN. By finding different minima over multiple optimisation starting points and penalty hyperparameter values, we build a posterior distribution over CRNs to quantify structural uncertainty. Some details can be found in `test/toy_example/README.md`.

The `src/` directory provides functionality for CRN data simulation, parameter inference, and network inference. See `test/` for applications: there is a simulation study (`toy_example/`) and two case studies based on experimental data (`pinene/` and `pyridine/`). The Jupyter notebook `crn_inference_tutorial.ipynb` features a standalone, simplified demo of CRN inference, without any uncertainty quantification. 

Parameter estimation is done in a frequentist fashion via maximising a penalised likelihood. To this end, we use a multi-start first-order optimisation algorithm, aided by automatic differentiation. A penalty function is used to encourage parsimonious model fits to the data. The implemented penalty functions are as follows:
- $L_1$ penalty on the original scale, i.e. an exponential prior,
- $L_1$ penalty on a shifted-log scale, i.e. a shifted-exponential prior on the log scale,
- approximate $L_0$ penalty, which is approximately proportional to the effective number of reactions *a la* Bayesian information criterion, 
- [horseshoe-like prior](https://arxiv.org/abs/1702.07400), which is a closed-form approximation of the horseshoe prior commonly used as a sparse prior in Bayesian hierarchical modelling.

Our approach shares some similarities to that of [Hoffmann et al. (2019)](https://doi.org/10.1063/1.5066099), who use a $L_1$ penalty (original scale). One notable difference is that Hoffmann et al. avoid numerically solving the ODE, and instead design a loss function based on derivative error, which relies on estimating derivatives from time series data. We also note that for our case study, the $L_1$ penalty (original scale) had the poorest overall performance out of the penalty functions considered.
