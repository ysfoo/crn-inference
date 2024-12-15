# Chemical reaction network inference

Project during MATRIX program "Parameter Identifiability in Mathematical Biology" (Sep 2024).

This repository provides Julia code for inferring the structure of a chemical reaction network (CRN) from noisy time series data of species concentrations. Our approach is based on constructing a system of ordinary differential equations (ODEs) from a library of candidate reactions, and subsequently performing parameter inference to estimate the reaction rate constants. Reactions with non-negligible rate constant estimates are deduced to be present in the network. 

The Jupyter notebook `crn_inference_tutorial.ipynb` (TODO) features a standalone demo of CRN inference. The `src/` directory provides functionality for CRN data simulation and parameter inference. The code and results for our case study are given in `test/`; see `test/README.md` for details.

Parameter estimation is done in a frequentist fashion via maximising a penalised likelihood. To this end, we use a multi-start first-order optimisation algorithm, aided by automatic differentiation. A penalty function is used to encourage parsimonious model fits to the data. We have implemented and compared the following penalty functions:
- $L_1$ penalty on the original scale, i.e. an exponential prior,
- $L_1$ penalty on a shifted-log scale, i.e. a shifted-exponential prior on the log scale,
- approximate $L_0$ penalty, which is approximately proportional to the effective number of reactions *a la* Bayesian information criterion, 
- [horseshoe-like prior](https://arxiv.org/abs/1702.07400), which is a closed-form approximation of the horseshoe prior commonly used as a sparse prior in Bayesian hierarchical modelling.

We also investigate whether it is beneficial to perform optimisation on the log scale of the reaction rate constants. For our case study, we found that abnormally large parameter estimates are often returned when optimisation is not performed on the log scale (with the exception of the $L_1$ penalty), indicating that the optimiser was stuck in some pathological local minimum. This corroborates with the benchmarking work of [Villaverde et al. (2018)](https://doi.org/10.1093/bioinformatics/bty736), who found that some parameter estimation problems can only be satisfactorily solved when parameters are log-transformed.

Our approach shares some similarities to that of [Hoffmann et al. (2019)](https://doi.org/10.1063/1.5066099), who use a $L_1$ penalty (original scale). One notable difference is that Hoffmann et al. avoid numerically solving the ODE, and instead design a loss function based on derivative error, which relies on estimating derivatives from time series data. We also note that for our case study, the $L_1$ penalty (original scale) had the poorest overall performance out of the penalty functions considered.

Deciding whether a reaction rate constant estimate is negligible or not is often done subjectively, e.g. in Hoffmann et al. (2019), a cutoff is manually chosen by visual inspection. We propose the following procedure for cutoff determination, which we implement in `src/inference.jl`:

1. Given $R$ candidate reactions with their rate constant estimates, sort the estimates in ascending order, and denote the result as $\hat{k}_{(1)}, \ldots, \hat{k}_{(R)}$.
2. Compute the ratios of consecutive rate constant estimates, i.e. $\hat{k}_{(j)} / \hat{k}_{(j-1)}$ for $j=2,\ldots,R$.
3. Suppose that $\hat{k}_{(j')} / \hat{k}_{(j'-1)}$ is the maximum of these ratios. The reactions corresponding to $\hat{k}_{(j')}, \hat{k}_{(j'+1)} \ldots, \hat{k}_{(R)}$ are deduced to be present in the network, i.e. the reaction with rate constant estimates at least $\hat{k}_{(j')}$.

For our case study, we find that this automated cutoff determination performs sufficiently well when the shifted-log scale $L_1$ penalty is used; see `test/vary_kvals/README.md`.
