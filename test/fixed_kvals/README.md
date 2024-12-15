## File descriptions

- `data.jl`: Script for simulating synthetic data based on the ground truth network; all reaction rate constants fixed at $1.0$.
- `workflow_fixed_opts.jl`: Standalone script that explains the steps leading up to parameter inference, including results for one optimisation instance ($L_1$ penalty, optimisation on log scale). This is designed to be run interactively line-by-line, e.g. in VS Code.
- `setup.jl`: Setup file included for convenience when multiple optimisation instances are involved.
- `inference_vary_opts.jl`: Script for running multiple instances of parameter inference for different optimisation settings, i.e. varying penalty functions, and whether optimisation is performed on the log space of the parameters or not.
- `plot_runs.jl`: Script for plotting results that include each optimisation run of each optimisation instance in `inference_vary_opts.jl`. The resulting images are stored in subdirectories of `output/`.
- `plot_eval.jl` [TODO]: Script for plotting results that summarise the best optimisation run for each optimisation instance in `inference_vary_opts.jl`. The resulting images are stored in `output/eval_figs/`.

## Choosing penalty hyperparameters

For the $L_1$ penalty and the horseshoe-like penalty, we tried a few hyperparameter values and found that $\lambda=20$ gave sufficiently good performance. This will soon be replaced by an automatic hyperparameter grid search. For the shifted-log $L_1$ penalty, we used $\lambda=1$ following [Gupta et al. (2020)](https://doi.org/10.1371/journal.pcbi.1007669). For the approximate $L_0$ penalty, we used $\lambda=\log(\text{no. of data points})$, a choice inspired by the Bayesian information criterion.

## Interpreting results

The images `inferred_trajs_run[x].png` show whether the trajectories reconstructed from estimated parameters follow the ground truth closely. The reconstruction is consistently accurate for all penalty functions when optimisation is performed on the log scale. Reconstruction quality is inconsistent when optimisation is performed on the log scale for all penalty functions except the $L_1$ penalty.

The images `inferred_rates_heatmap.png` summarise estimated reaction rate constants (clipped at 20.0, square root scale for colour). Note that the reactions in the ground truth network are outlined in boxes. For the current experiment, the $L_1$ penalty function results in the most consistent reconstruction of the ground truth rate constants (i.e. finding the global minimum). Under the other penalty functions, the optimiser sometimes finds local minima, which are especially poor when optimisation is not performed on the log scale. However, from experience, the good performance of the $L_1$ penalty function here is sensitive to the penalty hyperparameter and the ground truth rate constants &mdash; see `../vary_kvals/README.md`.

The images `inferred_rates_histogram.png` aggregate the estimated reaction rate constants over reactions and runs (y-axis clipped at 32). In order to infer which reactions are present in the system, we need to choose a cutoff value. The shifted-log $L_1$ penalty function leads to the clearest separation of negligible and non-negligible rate constants. Other penalty functions require a more subjective choice of a cutoff value, especially for the $L_1$ penalty function (on the original scale). This suggests that different penalty functions differ in the tradeoff between the ease of the finding the global minimum and a clear separation of negligible and non-negligible rate constants at the minimum found.

The images `inferred_rates_run[x].png` present a more detailed view of the estimated reaction rate constants (square root scale). Some of the local minima correspond to CRNs that are dynamically equivalent to the ground truth, e.g. the local minimum corresponding to `output/logL1_uselog/inferred_rates_run15.png` is explained by the fact that the reactions $X_3 \rightarrow X_1$ and $X_3 \rightarrow X_2 + X_3$ induce the same dynamics as the reaction $X_3 \rightarrow X_1 + X_2$ where when all these reactions share the same rate constants.