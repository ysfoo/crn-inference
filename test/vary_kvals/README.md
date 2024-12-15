We test the robustness of the penalty functions against different datasets simulated by varying the ground truth rate constants. Specifically, $k_1$ and $k_{18}$ take values from $\{0.1, 0.3, 1.0, 3.0, 10.0\}$, while $k_{13}$ is fixed at $1.0$. This results in $25$ datasets in total. 

Since the ideal hyperparameter values $\lambda$ may differ from the values chosen in `../fixed_kvals/`, we perform parameter inference with $\lambda = 0.5\lambda_\text{ref}, \lambda_\text{ref},$ or $2\lambda_\text{ref}$, where $\lambda_\text{ref}$ is the hyperparameter chosen in `../fixed_kvals/`. Given the results from `../fixed_vals/`, we always perform optimisation on the log scale.

## File descriptions

- `data.jl`: Script for simulating synthetic datasets based on the ground truth network with different rate constants.
- `setup.jl`: Setup file included for convenience when multiple optimisation instances are involved.
- `inference_vary_hyps.jl`: Script for running parameter inference for all datasets, while varying the penalty functions and hyperparameters.
- `plot_runs.jl`: Script for plotting results that include each optimisation run of each optimisation instance in `inference_vary_hyps.jl`. The resulting images are not stored in this repository to avoid bloating the repository size.
- `plot_eval.jl`: Script for plotting results that summarise the best optimisation run for each optimisation instance in `inference_vary_hyps.jl`. The resulting images are stored in `output/eval_figs/`.

## Interpreting results [TODO]