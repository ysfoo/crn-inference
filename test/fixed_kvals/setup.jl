############################################
### Setup used by inference and plotting ###
############################################

using DelimitedFiles

using Format
FMT_2DP = ".2f"; # `pyfmt(FMT_2DP, num)` converts a float `num` to a string with 2 decimal points

include(joinpath(@__DIR__, "../define_networks.jl")); # defines `full_network` and `k` (Symbolics object for rate constants)


# Known quantities
x0_map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # assume initial conditions are known
n_obs = 101; # number of time points
n_data = n_obs * length(x0_map) # number of data points
t_span = (0., 10.); # time interval to solve on
true_σs = vec(readdlm(joinpath(@__DIR__, "stds_true.txt")))

# Set up ODE based on CRN library (full network)
rx_vec = Catalyst.reactions(full_network); # list of reactions
n_rx = length(rx_vec);
oprob = ODEProblem(full_network, x0_map, t_span, zeros(n_rx)); # all rates here are zero, no dynamics

# Optimisation setup
LB, UB = 1e-10, 1e2; # bounds for reaction rate constants (i.e. parameters)
lbs = LB .* ones(n_rx);
ubs = UB .* ones(n_rx);

N_RUNS = 16; # number of optimisation runs

# All possible optimisation options
PEN_STRS = ["L1", "logL1", "approxL0", "hslike"]
HYP_VALS = 2.0 .^ (-3:6)
opt_options = collect(Iterators.product(PEN_STRS, HYP_VALS));

# Define ground truth
true_kvec = zeros(n_rx);
true_kvec[1] = true_kvec[18] = true_kvec[13] = 1.; # ground truth reaction rate constants


# Helper functions

# Read in data
function read_data(data_fname)
	fullmat = readdlm(data_fname);
	t_obs = fullmat[:,1];
	data = fullmat[:,2:end]';
	return t_obs, data # NB: `data` is a matrix of dimensions n_species * n_obs
end

# Directory name for where optimisation results are stored
function get_opt_dir(pen_str, hyp_val)
	return joinpath(@__DIR__, "output", pen_str, "hyp_$(hyp_val)")
end

