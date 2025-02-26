We test the robustness of the penalty functions against different datasets simulated by varying the ground truth rate constants. Specifically, $k_1$ and $k_{18}$ take values from $\{0.1, 0.3, 1.0, 3.0, 10.0\}$, while $k_{13}$ is fixed at $1.0$. This results in $25$ datasets in total. 

## File descriptions

- `data.jl`: Script for simulating synthetic datasets based on the ground truth network with different rate constants.
- `setup.jl`: Setup file included for convenience when multiple optimisation instances are involved.
- `inference_vary_kvals.jl`: Script for running parameter inference for all datasets, while varying the penalty functions and hyperparameters.
- `plot_runs.jl`: Script for plotting results that include each optimisation run of each optimisation instance in `inference_vary_kvals.jl`. The resulting images are stored in subdirectories of `output/`. The resulting images are not stored in this repository to avoid bloating the repository size.
- `evaluation.jl`: Script for plotting hyperparameter tuning results (stored in `output/vary_hyp_figs/`) and for plotting other evaluation results (stored in `output/eval_figs/`).

## Interpreting results

TODO