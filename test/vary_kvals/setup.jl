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

# Set up ODE based on CRN library (full network)
rx_vec = Catalyst.reactions(full_network); # list of reactions
n_rx = length(rx_vec);
oprob = ODEProblem(full_network, x0_map, t_span, zeros(n_rx)); # all rates here are zero, no dynamics

# Optimisation setup
LB, UB = 1e-10, 1e2; # bounds for reaction rate constants (i.e. parameters)
lbs = LB .* ones(n_rx);
ubs = UB .* ones(n_rx);

N_RUNS = 16; # number of optimisation runs

# All possible settings
K1_VALS = [0.1, 0.3, 1., 3., 10.];
K18_VALS = [0.1, 0.3, 1., 3., 10.];
PEN_STRS = ["L1", "logL1", "approxL0", "hslike"];
HYP_VALS = 2.0 .^ (-3:6);
settings_vec = collect(Iterators.product(
    K1_VALS, K18_VALS, PEN_STRS, HYP_VALS
));

# Helper functions

function read_data(data_fname)
	fullmat = readdlm(data_fname);
	t_obs = fullmat[:,1];
	data = fullmat[:,2:end]';
	return t_obs, data # NB: `data` is a matrix of dimensions n_species * n_obs
end

function get_data_dir(k1, k18)
    k1_str = pyfmt(".0e", k1)
    k18_str = pyfmt(".0e", k18)
    return joinpath(@__DIR__, "output/k1_$(k1_str)_k18_$(k18_str)")
end

function get_opt_dir(k1, k18, pen_str, hyp_val)
    data_dir = get_data_dir(k1, k18)
    joinpath(data_dir, pen_str, "hyp_$(hyp_val)")
end

function make_true_kvec(k1, k18)
    true_kvec = zeros(n_rx);
    true_kvec[1] = k1; true_kvec[18] = k18; true_kvec[13] = 1.;
    return true_kvec
end
