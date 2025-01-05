# Case study for reaction network inference

A reaction network is a directed graph with *complexes* as vertices and *reactions* as edges. In our case study, we consider

- 3 chemical species, namely $X_1, X_2, X_3$,
- 6 possible complexes, namely $X_1, X_2, X_3, X_1+X_2, X_2+X_3, X_1+X_3$, and
- $6\times 5 = 30$ possible reactions between distinct complexes; see `reactions.txt`.

We refer to the fully connected graph as the *library network* or the *full netowrk*.

 Reaction networks in real applications are much sparser than the fully connected graph. We use the following reaction network as the ground truth for our case study:

 ![equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}X_1\xrightarrow[]{k_{1}}X_2\quad\text{&space;and&space;}\quad&space;X_1&plus;X_2\overset{k_{18}}{\underset{k_{13}}\rightleftharpoons}X_3.)

Both the full network and the ground network are defined in `define_networks.jl`. We simulate data with additive Gaussian noise from the ground truth network at $N=101$ timepoints evenly spread across $[0,10]$. We then infer which reactions of the full network are present in the observed dynamics by minimising a loss function over all reaction rate constants (i.e. parameters), which consists of the negative log-likelihood and a penalty function. The penalty functions implemented here are:

- $L_1$ penalty on the original scale: $\mathrm{pen}(\theta) = \lambda \theta$,
- $L_1$ penalty on a shifted-log scale: $\mathrm{pen}(\theta) = \lambda \log\theta$,
- approximate $L_0$ penalty: $\mathrm{pen}(\theta) = \lambda \theta^{0.1}$,
- horseshoe-like penalty: $\mathrm{pen}(\theta) = -\log \log ( 1 + 1/(\lambda\theta)^2 )$,

where $\lambda$ is a hyperparameter to be chosen. See `penalty_funcs.png` for plots of these penalty functions. During optimisation, we impose a stricitly positive lower bound of $\theta \ge 10^{-10}$ as some penalty functions are undefined when $\theta=0$.

We use a multi-start optimisation approach, i.e. we execute multiple ($15$) optimisation runs each with different starting values for the parameters. Note that we assume that the initial conditions ($X_1(0)=X_2(0)=0, X_3(0)=1$) and noise standard deviations ($\sigma=0.01$) are known.

The directory `fixed_kvals/` investigates the case where the rate constants of the reactions in the ground truth network are all fixed to $1.0$. We evaluate the performance of different optimisation settings, that is, the four possible penalty functions, and whether optimisation is performed on the log space of the parameters or not. Note that we have hand-picked a reasonable hyperparameter for each penalty function. More details are given in `fixed_kvals/README.md`. 

The directory `vary_kvals/` investigates cases where the rate constants of the reactions in the ground truth network are varied. We evaluate the robustness of the different penalty functions against different datasets simulated from different ground truth rate constants. Additionally, we also adjust the penalty hyperparameters from the `fixed_kvals/` case to assess the sensitivity of each penalty function to its hyperparameter. More details are given in `fixed_kvals/README.md`. 

## Evaluation metrics

We define various evaluation metrics for trajectory reconstruction, parameter reconstruction, and network reconstruction, which are implemented in `evaluation.jl`.

- *Trajectory reconstruction.* Let $x(t)$ be a trajectory simulated from ground truth parameters and $\hat{x}(t)$ be the corresponding trajectory simulated from estimated parameters. We numerically compute $\max_t \lVert\hat{x}(t)-x(t)\rVert_1$ on some time grid as a measure of absolute trajectory reconstruction error.
- *Parameter reconstruction.* Let $k$ be the rate constant of a reaction from the ground truth network, and let $\hat{k}$ be the corresponding estimate. We use $\frac{\hat{k} - k}{k}$ as a relative parameter reconstruction error. A relative error of -1 corresponds to a reaction that was missed by inference (false negative).
- *Network reconstruction.* Suppose we have a decision rule that determines whether a reaction is present based on its estimated rate constant. We can then apply the usual evaluation metrics used in binary classification, e.g. precision and recall. Our decision rule is as follows: 

The metrics mentioned so far are most likely applied to the best optimisation run, i.e. the run which finds the lowest loss value. We define a basin of attraction metric as the proportion of runs that end with the same parameter values that result in this lowest loss value. Note that this lowest loss value is not guaranteed to be the global minimum. 

We define a loss offset to be a loss value subtracted by the loss value evaluated with the true parameter values. If the loss offset of the best run is positive, then by definition, the local minimum found during optimisation is not the global minimum. On the other hand, the true parameter values are not guaranteed to correspond to the global minimum. It is possible for the loss offset of the best run to be negative and for the corresponding inferred network to not match the ground truth. If this happens across all penalty functions for a problem instance, then this problem may suffer from an identifiability issue. However, if this happens only for a particular penalty function, then this may hint at weaknesses of that penalty function.

Finally, we are interested in how well each penalty function encourages sparsity. Given that our optimisation approach fixes $10^{-10}$ as a lower bound for the parameters, if a parameter is estimated to be less than $2\times 10^{-10}$, we will treat it as a zero. We define the sparsity fraction to be the proportion of reactions that have rate constant estimates less than $2\times 10^{-10}$ aggregated over all runs with a specific penalty function for a problem instance.

## Future work

Things to do:
- Finish computing all evaluation metrics
- Implement hyperparameter grid search
- Alternative cutoff determination (backward elimination)
- Reaction-specific hyperparameters
- Multicollinearity in reaction rates

Things unlikely to be done here but are likely needed in practice:
- Estimate noise SD (currently assumed to be known during inference)
- Estimate initial conditions (currently assumed to be known during inference)
- Handle multiple trajectories from different starting points
