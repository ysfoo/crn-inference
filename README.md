# Chemical reaction network inference

Project during MATRIX program "Parameter Identifiability in Mathematical Biology" (Sep 2024).

Run `sim_data.jl` to create synthetic data and run `full_network.jl` for network inference results.

The images `output/inferred_trajs_run[x].png` show that the trajectories reconstructed from estimated parameters follow the ground truth closely. However, parameter estimation is poor for runs 2, 4, 10 according to the images `output/inferred_rates_run[x].png`.