using Catalyst, OrdinaryDiffEq
using DelimitedFiles

t = default_t(); # time variable

# use labelling from Box et al. (1973)
# X1 = α-pinene, X2 = dipentene, X3 = allo-ocimene, X4 = pyronene, X5 = dimer
isomers = @species X1(t) X2(t) X3(t) X4(t);
dimer = (@species X5(t))[1];
species_vec = [isomers; dimer]

rxs_no_k = [
    [([reactant], [product]) for product in isomers, reactant in isomers if reactant !== product];
    [([dimer], [isomer], [1], [2]) for isomer in isomers];
    [([isomer], [dimer], [2], [1]) for isomer in isomers];
];
n_rx = length(rxs_no_k) # number of reactions
@parameters k[1:n_rx] # reaction rate constants
rx_vec = [
	Reaction(kval, rx_no_k...) for (rx_no_k, kval) in zip(rxs_no_k, k)
];

# CRN
@named full_network = ReactionSystem(rx_vec, t, species_vec, k; combinatoric_ratelaws=false)
full_network = complete(full_network)

# Initial conditions
x0_map = vcat([:X1 => 100.], [s.val.f.name => 0. for s in species_vec[2:end]])

# Read data
data_fname = joinpath(@__DIR__, "data.txt");
fullmat = readdlm(data_fname);
t_obs = fullmat[:,1];
data = fullmat[:,2:end]';
data[5,:] ./= 2.0; 
n_species, n_obs = size(data);

t_span = extrema(t_obs);
oprob = ODEProblem(full_network, x0_map, t_span, zeros(n_rx));

LB, UB = 1e-10, 1e2; # bounds for unscaled reaction rate constants (i.e. parameters)
lbs = LB .* ones(n_rx);
ubs = UB .* ones(n_rx);
σ_lbs, σ_ubs = fill(0.05, n_species), fill(5., n_species)

N_RUNS = 64; # number of optimisation runs

# All possible optimisation options
PEN_STRS = ["L1", "logL1", "approxL0", "hslike"]
HYP_VALS = 1:10
opt_options = collect(Iterators.product(PEN_STRS, HYP_VALS));

# Directory name for where optimisation results are stored
function get_opt_dir(pen_str, hyp_val)
	return joinpath(@__DIR__, "output", pen_str, "hyp_$(hyp_val)")
end