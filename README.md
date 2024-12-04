# Chemical reaction network (CRN) inference

Project during MATRIX program "Parameter Identifiability in Mathematical Biology" (Sep 2024).

Run `sim_data_script.jl` to create synthetic data. The ground truth CRN consists of the reactions

![equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}X_1\xrightarrow[]{k_{1}}X_2\quad\text{&space;and&space;}\quad&space;X_1&plus;X_2\overset{k_{18}}{\underset{k_{13}}\rightleftharpoons}X_3.)

The subscripts correspond to indices of a list of 30 candidate reactions as defined in `full_network.jl`; see `output/reactions.txt`. The default setup is to have all rate constants set to 1.0; these simulations are stored in `output/`. We also simulate other setups where the values $k_1$ and $k_{18}$ are varied; these simulations are stored in `output/vary_kvals/`.

Our aim is to infer the ground truth CRN from noisy time series data. This is done by estimating the rate constants of the 30 candidate reactions via a multi-start sparse optimisation approach. Each run involves starting the optimisation algorithm from a random initial value. The loss function to be optimised is `loss_func` as defined at the end of `inference.jl`. The key functions used for optimisation are implemented also in `inference.jl`. A demonstration of the usage of these functions is given in `inference_single.jl`, which is designed to be run interactively line-by-line, e.g. in VS Code.

## Varying optimisation options

We explore two optimisation options in `inference_vary_opts.jl`, with results stored in `output/vary_opts/`: (a) whether optimisation is performed on the log scale of the parameters, and (b) the penalty function of the parameters, which is one of
- $L_1$ penalty on the original scale, i.e. an exponential prior,
- $L_1$ penalty on a shifted-log scale, i.e. a shifted-exponential prior on the log scale,
- approximate $L_0$ penalty, which is approximately proportional to the effective number of reactions *a la* Bayesian information criterion, or
- [horseshoe-like prior](https://arxiv.org/abs/1702.07400), which is a closed-form approximation of the horseshoe prior commonly used as a sparse prior in Bayesian hierarchical modelling.

The images `inferred_trajs_run[x].png` show whether the trajectories reconstructed from estimated parameters follow the ground truth closely. The reconstruction is consistently accurate for all penalty functions when optimisation is performed on the log scale. Reconstruction quality is inconsistent when optimisation is performed on the log scale for all penalty functions except the $L_1$ penalty.

The images `inferred_rates_heatmap.png` summarise estimated reaction rate constants (clipped at 20.0, square root scale for colour). Note that the reactions in the ground truth network are outlined in boxes. For the current experiment, the $L_1$ penalty function results in the most consistent reconstruction of the ground truth rate constants (i.e. finding the global minimum). Under the other penalty functions, the optimiser sometimes finds local minima, which are especially poor when optimisation is not performed on the log scale. However, from experience, the good performance of the $L_1$ penalty function here is sensitive to the penalty hyperparameter and the ground truth rate constants &mdash; more sensitvity analysis is needed (for all penalty functions).

The images `inferred_rates_histogram.png` aggregate the estimated reaction rate constants over reactions and runs (y-axis clipped at 32). In order to infer which reactions are present in the system, we need to choose a cutoff value. The $L_1$ penalty function on the shifted-log scale leads to the clearest separation of negligible and non-negligible rate constants. Other penalty functions require a more subjective choice of a cutoff value, especially for the $L_1$ penalty function (on the original scale). This suggests that different penalty functions differ in the tradeoff between the ease of the finding the global minimum and a clear separation of negligible and non-negligible rate constants at the minima found.

The images `inferred_rates_run[x].png` present a more detailed view of the estimated reaction rates (square root scale). Some of the local minima correspond to CRNs that are dynamically equivalent to the ground truth, e.g. the local minimum corresponding to `output/vary_opts/logL1_uselog/inferred_rates_run15.png` is explained by the fact that the reactions $X_3 \rightarrow X_1$ and $X_3 \rightarrow X_2 + X_3$ induce the same dynamics as the reaction $X_3 \rightarrow X_1 + X_2$ where when all these reactions share the same rate constants.

## Varying penalty hyperparameters ground truth rate constants

The hyperparameters of the penalty functions have been manually chosen (to be documented). We test the robustness of our optimisation results against changes to the penalty hyperparameters by either halving or doubling them. This is done for 25 sets of ground truth rate constants, where $k_{13}$ is fixed at $1.0$, whereas $k_1$ and $k_{18}$ take values from $(0.1, 0.3, 1.0, 3.0, 10.0)$. Given the results from the previous section, we only perform optimisation in the log space of the parameters. 

Warning: There are currently around 10,000 images in this repository as a result of this sensitivity analysis. We need to find a more efficient way of summarising these results.

## Performance metrics

We need numerical measures of performance that reflect trajectory reconstruction, parameter reconstruction, and network reconstruction. Some of these results can be found in `eval_figs/`.

- *Trajectory reconstruction.* Let $x(t)$ be a trajectory on $t\in [0,T]$ simulated from ground truth parameters and $\hat{x}(t)$ be the corresponding trajectory simulated from estimated parameters. We numerically compute $\frac{1}{T}\int_0^T |\hat{x}(t)-x(t)| dt$ on some time grid as a measure of absolute trajectory reconstruction error.
- *Parameter reconstruction.* Let $k$ be the rate constant of a reaction from the true system, and let $\hat{k}$ be the corresponding estimate. We use $\frac{|\hat{k} - k|}{k}$ as a relative parameter reconstruction error.
- *Network reconstruction.* Suppose we have a classification rule that decides which reactions are present based on the estimated rate constants. We can then apply the metrics used in binary classification, e.g. precision and recall.

The metrics mentioned so far are most likely applied to the best optimisation run, i.e. the run which finds the lowest loss value. We define a basin of attraction metric as the proportion of runs that end with the same parameter values that result in this lowest loss value. Note that this lowest loss value is not guaranteed to be the global minimum. 

We define a loss offset to be a loss value subtracted by the loss value evaluated with the true parameter values. If the loss offset of the best run is positive, then by definition, the local minimum found during optimisation is not the global minimum. On the other hand, the true parameter values are not guaranteed to correspond to the global minimum. It is possible for the loss offset of the best run to be negative and for the corresponding inferred network to not match the ground truth. If this happens across all penalty functions for a problem instance, then this problem may suffer from an identifiability issue. However, if this happens only for a particular penalty function, then this may hint at weaknesses of that penalty function.

Finally, it would be interesting to note how well each penalty function encourages sparsity. Given that our optimisation approach fixes $10^{-10}$ as a lower bound for the parameters, if a parameter is estimated to be less than $2\times 10^{-10}$, we can essentially treat it as a zero. We define the sparsity fraction to be the proportion of reactions that have rate constant estimates less than $2\times 10^{-10}$ aggregated over all runs with a specific penalty function for a problem instance.

## Future work

Things to do:
- Summarise results
- Automatic determination of rate constant cutoff (use largest gap between estimated rate constants in log space as the cutoff)

Things I don't really want to do but are likely needed in practice:
- Estimate noise SD (currently assumed to be known during inference)
- Estimate initial conditions (currently assumed to be known during inference)
- Handle multiple trajectories from different starting points
- Automatic hyperparameter selection
