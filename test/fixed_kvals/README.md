We test our method for different penalty functions, using one dataset where all ground truth rate constants are $1.0$.

## File descriptions

- `data.jl`: Script for simulating synthetic data based on the ground truth network; all reaction rate constants are $1.0$.
- `workflow.jl`: Standalone script that explains the steps leading up to parameter inference, including results for one optimisation instance ($\log L_1$ penalty). This is designed to be run interactively line-by-line, e.g. in VS Code.
- `setup.jl`: Setup file included for convenience when multiple optimisation instances are involved.
- `inference_fixed_kvals.jl`: Script for running multiple instances of parameter inference for different penalty functions and hyperparameters.
- `plot_runs.jl`: Script for plotting results that include each optimisation run of each optimisation instance in `inference_fixed_kvals.jl`. The resulting images are stored in subdirectories of `output/`. The resulting images are not stored in this repository to avoid bloating the repository size.
- `evaluation.jl`: Script for plotting hyperparameter tuning results (stored in `output/vary_hyp_figs/`) and prints out the inferred reactions.
