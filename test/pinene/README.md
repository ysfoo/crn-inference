# Case study for $\alpha$-pinene isomerisation

This case study is described in Section 3.2 of the preprint.

## File descriptions

- `data.txt`: Data as tabulated in [Box et al. (1973)](https://doi.org/10.1080/00401706.1973.10489009).
- `setup.jl`: Setup file included to define the full network and corresponding ODE system.
- `inference_vary_hyp.jl`: Script for running parameter inference (preprint Section 2.2), while varying the penalty functions and hyperparameters.
- `refine_sols.jl`: Script for mapping parameter estimates to CRNs (preprint Section 2.3).
- `eval_all_runs.jl`: Script to generate results, including Figure 4, S4, Table 2, Section S3.2.
- `plot_tree.jl`: Script to generate Figure S5.