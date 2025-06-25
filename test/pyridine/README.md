# Case study for pyridine denitrogenation

This case study is described in Section 3.3 of the preprint. The carbon and nitrogen balance are imposed to generate the library of candidate reactions (Section S3.3).

## File descriptions

- `data.txt`: Data as tabulated in [Schittkowski (2009)](https://klaus-schittkowski.de/ds_test_problems.pdf).
- `setup.jl`: Setup file included to define the full network and corresponding ODE system.
- `inference_vary_hyp.jl`: Script for running parameter inference (preprint Section 2.2), while varying the penalty functions and hyperparameters.
- `refine_sols.jl`: Script for mapping parameter estimates to CRNs (preprint Section 2.3).
- `eval_all_runs.jl`: Script to generate results, including Figure 5 and S6, Section S3.3.
- `plot_tree.jl`: Script to generate Figure S7.