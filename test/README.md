# Simulation study for reaction network inference

A reaction network is a directed graph with *complexes* as vertices and *reactions* as edges. In our case study, we consider

- 3 chemical species, namely $X_1, X_2, X_3$,
- 6 possible complexes, namely $X_1, X_2, X_3, X_1+X_2, X_2+X_3, X_1+X_3$, and
- $6\times 5 = 30$ possible reactions between distinct complexes; see `reactions.txt`.

We refer to the fully connected graph as the *library network* or the *full netowrk*.

 Reaction networks in real applications are much sparser than the fully connected graph. We use the following reaction network as the ground truth for our case study:

 ![equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}X_1\xrightarrow[]{k_{1}}X_2\quad\text{&space;and&space;}\quad&space;X_1&plus;X_2\overset{k_{18}}{\underset{k_{13}}\rightleftharpoons}X_3.)

Both the full network and the ground network are defined in `define_networks.jl`. We simulate data with additive Gaussian noise (standard deviation $=0.01$) from the ground truth network at $N=101$ timepoints evenly spread across $[0,10]$. We then infer which reactions of the full network are present in the observed dynamics by minimising a loss function over all reaction rate constants, which consists of the negative log-likelihood and a penalty function. The penalty functions implemented here are:

- $L_1$ penalty on the original scale: $\mathrm{pen}(\theta) = \lambda \theta$,
- $L_1$ penalty on a shifted-log scale: $\mathrm{pen}(\theta) = \lambda \log\theta$,
- approximate $L_0$ penalty: $\mathrm{pen}(\theta) = \lambda \theta^{0.1}$,
- horseshoe-like penalty: $\mathrm{pen}(\theta) = -\log \log ( 1 + 1/(\lambda\theta)^2 )$,

where $\lambda$ is a hyperparameter to be chosen, and $\theta$ is a normalised (see `../details.md`) rate constant. See `penalty_funcs.png` for plots of these penalty functions. During optimisation, we impose a stricitly positive lower bound of $\theta \ge 10^{-10}$ as some penalty functions are undefined when $\theta=0$. We optimise over the noise standard deviations (assumed unknown) and log-transformed normalised rate constants.

We use a multi-start optimisation approach, i.e. we execute multiple optimisation runs each with different starting values for the parameters (noise standard deviations and normalised rate constants). Note that we assume that the initial conditions ($X_1(0)=X_2(0)=0, X_3(0)=1$) are known. 

The directory `fixed_kvals/` investigates the case where the rate constants of the reactions in the ground truth network are all fixed to $1.0$. We compare the results of using different penalty functions. More details are given in `fixed_kvals/README.md`. 

The directory `vary_kvals/` investigates cases where the rate constants of the reactions in the ground truth network are varied. We evaluate the robustness of the different penalty functions against different datasets simulated from different ground truth rate constants. More details are given in `fixed_kvals/README.md`. 

## Hyperparameter tuning

TODO

## Evaluation metrics

We define various evaluation metrics for trajectory reconstruction, parameter reconstruction, and network reconstruction, which are implemented in `evaluation.jl`.

- *Network reconstruction.* We have implemented an automatic procedure for determining, from rate constant estimates, which reactions present in the network (see `../details.md). We can then evaluate how many ground-truth reactions are correctly inferred (true positives), and how many reactions are spuriously inferred (false positives).
- *Trajectory reconstruction.* Let $x(t)$ be a trajectory simulated from ground truth parameters and $\hat{x}(t)$ be the corresponding trajectory simulated from estimated parameters. We numerically compute $\max_t \lVert\hat{x}(t)-x(t)\rVert_1$ on some time grid as a measure of absolute trajectory reconstruction error.
- *Parameter reconstruction.* Let $k$ be the rate constant of a reaction from the ground truth network, and let $\hat{k}$ be the corresponding estimate. We use $\frac{\hat{k} - k}{k}$ as a relative parameter reconstruction error. A relative error of -1 corresponds to a reaction that was missed by inference (false negative).

We are also interested in how well each penalty function encourages sparsity. Given that our optimisation approach fixes $10^{-10}$ as a lower bound for the normalised rate constants, if a normalised rate constant is estimated to be less than $2\times 10^{-10}$, we will treat it as a zero. We define the sparsity fraction to be the proportion of reactions that have rate constant estimates less than $2\times 10^{-10}$ aggregated over all runs with a specific penalty function for a problem instance.

## Interpreting results from `make_plots_runs`

The images `inferred_rates_heatmap.png` summarise estimated reaction rate constants (clipped at 20.0, square root scale for colour). Note that the reactions in the ground truth network are outlined in boxes. This gives a sense of how many local minima the optimisation algorithm finds. For large hyperparameter values, the penalised loss function induced by the $L_1$ penalty has a very large basin of attraction for its global minimum. The same is not true for the other penalty functions.

The images `inferred_rates_histogram.png` aggregate the estimated reaction rate constants over reactions and runs (y-axis clipped at 32). The shifted-log $L_1$ penalty function leads to the clearest separation of negligible and non-negligible rate constants (normalised). Other penalty functions do not provide a clear separation, especially the $L_1$ penalty function.

The last two sets of images are only produced when argument `indiv_runs` is set to `true` in the function `make_plots_runs`.

The images `inferred_trajs_run[x].png` show whether the trajectories reconstructed from estimated parameters follow the ground truth closely. 

The images `inferred_rates_run[x].png` present a more detailed view of the estimated reaction rate constants (square root scale). Some of the local minima correspond to CRNs that are [dynamically equivalent](https://reaction-networks.net/wiki/Dynamical_equivalence) to the ground truth.