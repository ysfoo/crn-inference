# Chemical reaction network (CRN) inference

Project during MATRIX program "Parameter Identifiability in Mathematical Biology" (Sep 2024).

Run `sim_data_script.jl` to create synthetic data. The ground truth CRN consists of the reactions

![equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}X_1\xrightarrow[]{k_{1}}X_2\quad\text{&space;and&space;}\quad&space;X_1&plus;X_2\overset{k_{18}}{\underset{k_{13}}\rightleftharpoons}X_3.)

The subscripts correspond to indices of a list of 30 candidate reactions as defined in `full_network.jl`; see `output/reactions.txt`. The default setup is to have all rate constants set to 1.0; these simulations are stored in `output/`. We also simulate other setups where the values $k_1$ and $k_{18}$ are varied; these simulations are stored in `output/vary_kvals/`.

Our aim is to infer the ground truth CRN from noisy time series data. This is done by estimating the rate constants of the 30 candidate reactions via a multi-start sparse optimisation approach. The key functions are implemented in `inference.jl`. A demonstration of the usage of these functions is given in `inference_single.jl`, which is designed to be run interactively line-by-line, e.g. in VS Code.

## Varying optimisation options

We explore two optimisation options in `inference_vary_opts.jl`, with results stored in `output/vary_opts/`: (a) whether optimisation is performed on the log scale of the parameters, and (b) the penalty function of the parameters, which is one of
- $L_1$ penalty on the original scale, i.e. an exponential prior,
- $L_1$ penalty on a shifted-log scale, i.e. a shifted-exponential prior on the log scale,
- approximate $L_0$ penalty, which is approximately proportional to the effective number of reactions *a la* Bayesian information criterion, or
- [horseshoe-like prior](https://arxiv.org/abs/1702.07400), which is a closed-form approximation of the horseshoe prior commonly used as a sparse prior in Bayesian hierarchical modelling.

The images `inferred_trajs_run[x].png` show whether the trajectories reconstructed from estimated parameters follow the ground truth closely. The reconstruction is consistently accurate for all penalty functions when optimisation is performed on the log scale. Reconstruction quality is inconsistent when optimisation is performed on the log scale for all penalty functions except the $L_1$ penalty.

The images `inferred_rates_heatmap.png` summarise estimated reaction rate constants. Note that the reactions in the ground truth network are outlined in boxes. For the current experiment, the $L_1$ penalty function results in the most consistent reconstruction of the ground truth rate constants (i.e. finding the global minimum). Under the other penalty functions, the optimiser sometimes finds local minima, which are especially poor when optimisation is not performed on the log scale. However, from experience, the good performance of the $L_1$ penalty function here is sensitive to the penalty hyperparameter and the ground truth rate constants &mdash; more sensitvity analysis is needed (for all penalty functions).

The images `inferred_rates_histogram.png` aggregate the estimated reaction rate constants over reactions and runs. In order to infer which reactions are present in the system, we need to choose a cutoff value. The $L_1$ penalty function on the shifted-log scale leads to the clearest separation of negligible and non-negligible rate constants. Other penalty functions require a more subjective choice of a cutoff value, especially for the $L_1$ penalty function (on the original scale). This suggests that different penalty functions differ in the tradeoff between the ease of the finding the global minimum and a clear separation of negligible and non-negligible rate constants at the minima found.

The images `inferred_rates_run[x].png` present a more detailed view of the estimated reaction rates. Some of the local minima correspond to CRNs that are dynamically equivalent to the ground truth, e.g. the local minimum corresponding to `output/vary_opts/logL1_uselog/inferred_rates_run15.png` is explained by the fact that the reactions $X_3 \rightarrow X_1$ and $X_3 \rightarrow X_2 + X_3$ induce the same dynamics as the reaction $X_3 \rightarrow X_1 + X_2$ where when all these reactions share the same rate constants.

## Future work

Things to do:
- Automatic determination of reaction rate cutoff
- Sensitivity to penalty hyperparameter
- Robustness of results against a variety of ground truth reaction rate constants

Things I don't really want to do but are likely needed in practice:
- Estimate noise SD (currently assumed to be known during inference)
- Estimate initial conditions (currently assumed to be known during inference)
- Handle multiple trajectories from different starting points
- Automatic hyperparameter selection
