########################################################
### Simulate data where all rate constants are fixed ###
########################################################

using StableRNGs
include(joinpath(@__DIR__, "../../src/sim_data.jl")); # imports `sim_data` for simulating data
include(joinpath(@__DIR__, "../define_networks.jl")); # defines true_rn

true_kmap  = (:k1 => 1., :k18 => 1., :k13 => 1.); # reaction rate constants
true_x0_map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # initial conditions
t_span = (0., 10.); # time interval to solve on
n_obs = 101;
σ_func(x) = 0.01*(maximum(x)-minimum(x)); # noise std = 5% of range

rng = StableRNG(1); # random seed for reproducibility
data_dir = @__DIR__; # data will be stored in same directory as this file, including plots
sim_data(true_rn, true_kmap, true_x0_map, t_span, n_obs, σ_func, data_dir, rng);
