############################################
### Setup used by inference and plotting ###
############################################

using DelimitedFiles
using Random

using Format
FMT_2DP = ".2f"; # `pyfmt(FMT_2DP, num)` converts a float `num` to a string with 2 decimal points

include(joinpath(@__DIR__, "../define_networks.jl")); # defines `full_network` and `k` (Symbolics object for rate constants)

# Known quantities
x0map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # assume initial conditions are known
n_obs = 101; # number of time points
n_data = n_obs * length(x0map) # number of data points
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

# All possible settings
k1_choice = [0.1, 0.3, 1., 3., 10.];
k18_choice = [0.1, 0.3, 1., 3., 10.];
pen_choice = ["L1", "logL1", "approxL0", "hslike"]; # penalty_function
hyp_choice = ["half_hyps", "orig_hyps", "double_hyps"] # hyperparameter values (relative to default)
settings_vec = collect(Iterators.product(
    k1_choice, k18_choice, pen_choice, hyp_choice
));

# Default hyperparameter values
HYP_DICT = Dict(
	"L1" => 20.0, # manually chosen
	"logL1" => 1.0, # taken from Gupta et al. (2020)
	"approxL0" => log(n_data), # BIC-inspired
	"hslike" => 20.0 # manually chosen
);

# Multipliers for varying hyperparameter values
MULT_DICT = Dict("half_hyps" => 0.5, "orig_hyps" => 1.0, "double_hyps" => 2.0);

# Helper functions

function read_data(data_dir, data_fname="data.txt")
	fullmat = readdlm(joinpath(data_dir, data_fname));
	t_obs = fullmat[:,1];
	data = fullmat[:,2:end]';
	return t_obs, data # NB: `data` is a matrix of dimensions n_species * n_obs
end

function get_data_dir(k1, k18)
    k1_str = pyfmt(".0e", k1)
    k18_str = pyfmt(".0e", k18)
    return joinpath(@__DIR__, "output/k1_$(k1_str)_k18_$(k18_str)")
end

function get_opt_dir(k1, k18, pen_str, hyp_str)
    data_dir = get_data_dir(k1, k18)
    joinpath(data_dir, hyp_str, pen_str * "_uselog")
end

function make_true_kvec(k1, k18)
    true_kvec = zeros(n_rx);
    true_kvec[1] = k1; true_kvec[18] = k18; true_kvec[13] = 1.;
    return true_kvec
end

function get_hyp(pen_str, hyp_str)
    return HYP_DICT[pen_str] * MULT_DICT[hyp_str]
end
