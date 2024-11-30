# Chemical reaction network (CRN) inference

Project during MATRIX program "Parameter Identifiability in Mathematical Biology" (Sep 2024).

Run `sim_data.jl` to create synthetic data. The ground truth CRN consists of the reactions

![equation](https://latex.codecogs.com/svg.image?%5Cbg_white%20X_1\xrightarrow[]{k_{1}}X_2\quad\text{&space;and&space;}\quad&space;X_1&plus;X_2\overset{k_{18}}{\underset{k_{13}}\rightleftharpoons}X_3.)

The subscripts correspond to indices of a list of 30 candidate reactions; see `output/reactions.txt`. Our aim is to infer the ground truth CRN from noisy time series data generated from `sim_data.jl`. This is done by estimating the rate constants of the 30 candidate reactions via a sparse optimisation approach.

Run `inference.jl` for CRN inference results. The script `inference.jl` requires command-line arguments to be specified; run `julia inference.jl --help` for a description. In summary, these command-line arguments specify (a) whether optimisation is performed on the log scale of the parameters, and (b) the penalty function of the parameters, which is one of
- $L_1$ penalty on the original scale, i.e. an exponential prior,
- $L_1$ penalty on a shifted-log scale, i.e. a shifted-exponential prior on the log scale,
- approximate $L_0$ penalty, which is proportional to the effective number of reactions *a la* Bayesian information criterion, or
- [horseshoe-like prior](https://arxiv.org/abs/1702.07400), which is a closed-form approximation of the horseshoe prior commonly used as a sparse prior in Bayesian hierarchical modelling.

The results under `output` are split across subdirectories corresponding to these optimisation options.

The images `inferred_trajs_run[x].png` show whether the trajectories reconstructed from estimated parameters follow the ground truth closely. The reconstruction is consistently accurate for all penalty functions when optimisation is performed on the log scale. Reconstruction quality is inconsistent when optimisation is performed on the log scale for all penalty functions except the $L_1$ penalty.

The images `inferred_rates_heatmap.png` summarise estimated reaction rate constants. Note that the reactions in the ground truth network are outlined in boxes. For the current experiment, the $L_1$ penalty function results in the most consistent reconstruction of the ground truth rate constants. Under the other penalty functions, the optimiser sometimes finds local minima, which are especially poor when optimisation is not performed on the log scale. However, from experience, the good performance of the $L_1$ penalty function here is sensitive to the penalty hyperparameter and the ground truth rate constants &mdash; more sensitvity analysis is needed (for all penalty functions).

The images `inferred_rates_histogram.png` aggregate the estimated reaction rate constants over reactions and runs. In order to infer which reactions are present in the system, we need to choose a cutoff value. The $L_1$ penalty function on the shifted-log scale leads to the clearest separation of negligible and non-negligible rate constants. Other penalty functions require a more subjective choice of a cutoff value.

The images `inferred_rates_run[x].png` present a more detailed view of the estimated reaction rates. Some of the local minima correspond to CRNs that are dynamically equivalent to the ground truth, e.g. the local minimum corresponding to `output/logL1_uselog/inferred_rates_run3.png` is explained by the fact that the reactions $X_3 \rightarrow X_2$ and $X_3 \rightarrow X_1 + X_3$ induce the same dynamics as the reaction $X_3 \rightarrow X_1 + X_2$ where when all these reactions share the same rate constants.

Things to do if I can be bothered to:
- Automatic determination of reaction rate cutoff
- Sensitivity to penalty hyperparameter
- Robustness of results against a variety of ground truth reaction rate constants

Things I don't really want to do but are likely needed in practice:
- Estimate noise SD (currently assumed to be known during inference)
- Estimate initial conditions (currently assumed to be known during inference)
- Handle multiple trajectories from different starting points
- Automatic hyperparameter selection
