# Simulation study for CRN inference

A reaction network is a directed graph with *complexes* as vertices and *reactions* as edges. In our case study, we consider

- 3 chemical species, namely $X_1, X_2, X_3$,
- 6 possible complexes, namely $X_1, X_2, X_3, X_1+X_2, X_2+X_3, X_1+X_3$, and
- $6\times 5 = 30$ possible reactions between distinct complexes; see `reactions.txt`.

We refer to the fully connected graph as the *library network* or the *full netowrk*.

Reaction networks in real applications are much sparser than the fully connected graph. We use the following reaction network as the ground truth for our case study:

 ![equation](https://latex.codecogs.com/png.image?\dpi{110}\bg{white}X_1\xrightarrow[]{k_{1}}X_2\quad\text{&space;and&space;}\quad&space;X_1&plus;X_2\overset{k_{18}}{\underset{k_{13}}\rightleftharpoons}X_3.)

Both the full network and the ground-truth network are defined in `define_networks.jl`. We simulate data with additive Gaussian noise (standard deviation is 1% of the true range) from the ground truth network at $N=101$ timepoints evenly spread across $[0,10]$. Observations are clamped to be nonnegative. We then infer which reactions of the full network are present in the observed dynamics by minimising a loss function over all reaction rate constants, which consists of the negative log-likelihood and a penalty function. The penalty functions implemented here are:

- $L_1$ penalty on the original scale: $\mathrm{pen}(k) = \lambda k$,
- $L_1$ penalty on a shifted-log scale: $\mathrm{pen}(k) = \lambda \log k$,
- Approximate $L_0$ penalty: $\mathrm{pen}(k) = \lambda k^{0.1}$,
- Horseshoe-like penalty: $\mathrm{pen}(k) = -\log \log ( 1 + 1/(\lambda k)^2 )$,

where $\lambda$ is a hyperparameter to be chosen, and $k$ is a normalised rate constant (see preprint Section S1.2). See `../penalty_funcs.png` for plots of these penalty functions. During optimisation, we impose a stricitly positive lower bound of $k \ge 10^{-10}$ as some penalty functions are undefined when $k=0$. We optimise over the noise standard deviations (assumed unknown) and log-transformed normalised rate constants.

We use a multi-start optimisation approach, i.e. we execute multiple optimisation runs each with different starting values for the parameters (noise standard deviations and normalised rate constants). Note that we assume that the initial conditions ($X_1(0)=X_2(0)=0, X_3(0)=1$) are known. 

We test the robustness of the penalty functions against different datasets simulated by varying the ground truth rate constants. Specifically, $k_1$ and $k_{18}$ take values from $\{0.1, 0.3, 1.0, 3.0, 10.0\}$, while $k_{13}$ is fixed at $1.0$. This results in $25$ datasets in total. 

## File descriptions

- `data.jl`: Script for simulating synthetic datasets based on the ground truth network with different rate constants.
- `setup.jl`: Setup file included for convenience when multiple optimisation instances are involved.
- `inference_vary_kvals.jl`: Script for running parameter inference (preprint Section 2.2) for all datasets, while varying the penalty functions and hyperparameters.
- `refine_sols.jl`: Script for mapping parameter estimates to CRNs (preprint Section 2.3).
- `eval_all_runs.jl`: Script to generate results, including Figures 1, 2, S1, S2, S3, Table 1, Section S3.1.
- `uncertainty_example.jl`: Script to generate Figure 3.