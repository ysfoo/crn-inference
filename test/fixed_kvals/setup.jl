############################################
### Setup used by inference and plotting ###
############################################

using DelimitedFiles
using Random

using Format
FMT_2DP = ".2f"; # `pyfmt(FMT_2DP, num)` converts a float `num` to a string with 2 decimal points

include(joinpath(@__DIR__, "../define_networks.jl")); # defines `full_network` and `k` (Symbolics object for rate constants)


# Helper functions

# Read in data
function read_data(data_dir, data_fname="data.txt")
	fullmat = readdlm(joinpath(data_dir, data_fname));
	t_obs = fullmat[:,1];
	data = fullmat[:,2:end]';
	return t_obs, data # NB: `data` is a matrix of dimensions n_species x n_obs
end

# Directory name for where optimisation results are stored
function get_opt_dir(pen_str, log_opt)
	return joinpath(@__DIR__, "output", pen_str * "_" * (log_opt ? "uselog" : "nolog"))
end


# Import synthetic data
t_obs, data = read_data(joinpath(@__DIR__, "output"));
n_obs = length(t_obs);

# Known quantities
x0map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # assume initial conditions are known
t_span = (0., 10.); # time interval to solve on
σ = 0.01; # assume noise is known

# Set up ODE based on CRN library (full network)
rx_vec = Catalyst.reactions(full_network); # list of reactions
n_rx = length(rx_vec);
oprob = ODEProblem(full_network, x0map, t_span, zeros(n_rx)); # all rates here are zero, no dynamics

# Optimisation setup
LB, UB = 1e-10, 1e2; # bounds for reaction rate constants (i.e. parameters)
lbs = LB .* ones(n_rx);
ubs = UB .* ones(n_rx);

N_RUNS = 15; # number of optimisation runs

# All possible optimisation options
opt_options = Iterators.product(["L1", "logL1", "approxL0", "hslike"], [true, false]);

# Default hyperparameter values
HYP_DICT = Dict(
	"L1" => 20.0, # manually chosen
	"logL1" => 1.0, # taken from Gupta et al. (2020)
	"approxL0" => log(length(data)), # BIC-inspired
	"hslike" => 20.0 # manually chosen
);

# Define ground truth
true_kvec = zeros(n_rx);
true_kvec[1] = true_kvec[18] = true_kvec[13] = 1.; # ground truth reaction rate constants