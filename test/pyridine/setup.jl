using Catalyst, OrdinaryDiffEq
using DelimitedFiles

include(joinpath(@__DIR__, "../../src/inference.jl"));

t = default_t(); # time variable

# use labelling from Bock (1981)
# X1 = pyridine, X2 = piperidine, X3 = pentylamine, X4 = N-pentylpiperidine, X5 = dipentylamine, X6 = ammonia, X7 = pentane
species_vec = @species X1(t) X2(t) X3(t) X4(t) X5(t) X6(t);
pentane = (@species X7(t))[1];
prds_vec = [species_vec; pentane];
carbon_dict = Dict(X1 => 5, X2 => 5, X3 => 5, X4 => 10, X5 => 10, X6 => 0, X7 => 5)
nitrogen_dict = Dict(X1 => 1, X2 => 1, X3 => 1, X4 => 1, X5 => 1, X6 => 1, X7 => 0)

# Reactant complexes
prd_cpxs = [
    [([s], [1]) for s in prds_vec]; 
    [([s], [2]) for s in prds_vec]; 
    [([prds_vec[i], prds_vec[j]], [1, 1]) for i in 1:length(prds_vec) for j in 1:length(prds_vec) if i < j]
]
prd_to_c = Dict((prds, stoichs) => sum(stoichs.*get.(Ref(carbon_dict),prds,0)) for (prds, stoichs) in prd_cpxs)
prd_to_n = Dict((prds, stoichs) => sum(stoichs.*get.(Ref(nitrogen_dict),prds,0)) for (prds, stoichs) in prd_cpxs)
rct_cpxs = [(prds, stoichs) for (prds, stoichs) in prd_cpxs if all(p.val.f.name != :X7 for p in prds)]

function no_overlap(xs, ys)
    return all(x !== y for x in xs, y in ys)
end

rxs_no_k = [
    (r, p) for p in prd_cpxs, r in rct_cpxs if prd_to_c[r] == prd_to_c[p] && prd_to_n[r] == prd_to_n[p] && no_overlap(r[1], p[1])
];
n_rx = length(rxs_no_k) # number of reactions
@parameters k[1:n_rx] # reaction rate constants

function remove_pentane((x, stoich))
    is_pentane(s) = s.val.f.name == :X7
    return (x[.!is_pentane.(x)], stoich[.!is_pentane.(x)])
end

rx_vec = [
	begin
        rct, rct_stoich = remove_pentane(rct_tuple);
        prd, prd_stoich = remove_pentane(prd_tuple);
        Reaction(kval, rct, prd, rct_stoich, prd_stoich) 
    end for ((rct_tuple, prd_tuple), kval) in zip(rxs_no_k, k)
];

# CRN
@named full_network = ReactionSystem(rx_vec, t, species_vec, k; combinatoric_ratelaws=false)
full_network = complete(full_network)

# Initial conditions
x0_map = vcat([:X1 => 1.], [s.val.f.name => 0. for s in species_vec[2:end]])

# Read data
data_fname = joinpath(@__DIR__, "data.txt");
fullmat = readdlm(data_fname);
t_obs = fullmat[:,1];
data = fullmat[:,2:end]';
n_species, n_obs = size(data);

t_span = extrema(t_obs);
t_grid = range(t_span..., 500)
oprob = ODEProblem(full_network, x0_map, t_span, zeros(n_rx));

alg = AutoVern7(KenCarp4());
abstol = 1e-10;

# Smooth data
smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
init_σs = [estim_σ(datarow .- eval_spline(res..., t_obs)) for (datarow, res) in zip(eachrow(data), smooth_resvec)]
scale_fcts = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)

LB, UB = 1e-10, 1e2; # bounds for unscaled reaction rate constants (i.e. parameters)
lbs = LB .* ones(n_rx);
ubs = UB .* ones(n_rx);
σ_lbs, σ_ubs = fill(1e-5, n_species), fill(1e-1, n_species)

N_RUNS = 256; # number of optimisation runs

# All possible optimisation options
PEN_STRS = ["L1", "logL1", "approxL0", "hslike"]
HYP_VALS = 1:10
opt_options = collect(Iterators.product(PEN_STRS, HYP_VALS));

# Directory name for where optimisation results are stored
function get_opt_dir(pen_str, hyp_val)
	return joinpath(@__DIR__, "output", pen_str, "hyp_$(hyp_val)")
end

# Gold standard
true_rx_tuples = [
    (([X1], [1]), ([X2], [1])),
    (([X2], [1]), ([X3], [1])),
    (([X2,X3], [1,1]), ([X4,X6], [1,1])),
    (([X3], [2]), ([X5,X6], [1,1])),
    (([X4], [1]), ([X5], [1])),
    (([X3], [1]), ([X6,X7], [1,1])),
    (([X4], [1]), ([X2,X7], [1,1])),
    (([X5], [1]), ([X3,X7], [1,1])),
    (([X2], [1]), ([X1], [1])),
    (([X4,X6], [1,1]), ([X2,X3], [1,1])),
    (([X5,X6], [1,1]), ([X3], [2]))
];
function match_rxs(x, y)
    all(all(x[i][1] .=== y[i][1]) && all(x[i][2] .== y[i][2]) for i in 1:2)
end
gold_idxs = [findfirst((x)->match_rxs(x,rx_tuple), rxs_no_k) for rx_tuple in true_rx_tuples]

n_gold_rx = length(gold_idxs) # number of reactions