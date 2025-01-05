########################################################
### Simulate data where all rate constants are fixed ###
########################################################

include(joinpath(@__DIR__, "../../src/sim_data.jl")); # imports `sim_data` for simulating data
include(joinpath(@__DIR__, "../define_networks.jl")); # defines true_rn

true_kmap  = (:k1 => 1., :k18 => 1., :k13 => 1.); # reaction rate constants
true_x0map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # initial conditions
t_span = (0., 10.); # time interval to solve on
n_obs = 101;
σ = 0.01;

data_dir = @__DIR__;
mkpath(data_dir); # create directory

sim_data(
	true_rn, true_kmap, true_x0map, t_span, n_obs, σ,
	data_dir, 2024
);