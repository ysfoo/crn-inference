# Chemical reaction network inference

Project during MATRIX program "Parameter Identifiability in Mathematical Biology" (Sep 2024).

Run `sim_data.jl` to create synthetic data and run `full_network.jl` for network inference results.

The images `output/inferred_trajs_run[x].png` show that the trajectories reconstructed from estimated parameters follow the ground truth closely. However, parameter estimation is poor for runs 2, 4, 10 according to the images `output/inferred_rates_run[x].png`. A limitation is that the determination of whether a reaction is present depends on a subjective cutoff of the reaction rate constant.

Things to do if I can be bothered to:
- Automatic determination of reaction rate cutoff
- Repeat for a variety of ground truth reaction rate constants for robustness of results
- Experiment with different penalty functions
- Automatic selection of penalty hyperparameter (e.g. the 10 in Exponential(rate = 10))

Things I don't really want to do but are likely needed in practice:
- Estimate noise SD
- Estimate initial conditions
- Handle multiple trajectories from different starting points