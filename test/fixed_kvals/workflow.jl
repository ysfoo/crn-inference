###################################################################
### Perform inference for a single instance and produce results ###
###################################################################

using DelimitedFiles
using StableRNGs

using Format
FMT_2DP = ".2f"; # `pyfmt(FMT_2DP, num)` converts a float `num` to a string with 2 decimal points

include(joinpath(@__DIR__, "../define_networks.jl")); # defines `full_network` and `k` (Symbolics object for rate constants)
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference
include(joinpath(@__DIR__, "../eval_helper.jl")); # imports functions used for results


### Setup
x0map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # assume initial conditions are known
t_span = (0., 10.);
rx_vec = Catalyst.reactions(full_network); # list of reactions
n_rx = length(rx_vec);
oprob = ODEProblem(full_network, x0map, t_span, zeros(n_rx)); # all rates here are zero, no dynamics

LB, UB = 1e-10, 1e2; # bounds for reaction rate constants (i.e. parameters)
lbs = LB .* ones(n_rx);
ubs = UB .* ones(n_rx);


### Import synthetic data
data_fname = joinpath(@__DIR__, "data.txt");
fullmat = readdlm(data_fname);
t_obs = fullmat[:,1];
data = fullmat[:,2:end]';
n_species, n_obs = size(data);

# Verify full CRN can reproduce 'ground truth' CRN
true_kvec = zeros(n_rx);
true_kvec[1] = true_kvec[18] = true_kvec[13] = 1.;
sol = solve(remake(oprob, p=[k => true_kvec]));

sol.u[end] # final point for trajectory simulated from full model
data[:,end] # final data point (noisy), should be similar to above


### Data preprocessing
smoothers = smooth_data.(eachrow(data), Ref(t_obs));

# visualise smoothers
f = Figure();
ax = Axis(f[1,1]);
for i in 1:n_species
    lines!(smoothers[i].t̂, smoothers[i].û)
    scatter!(t_obs, data[i,:], alpha=0.6)
end
current_figure()

σs_init = estim_σ.(eachrow(data), smoothers)
scale_fcts = get_scale_fcts(smoothers, species_vec, rx_vec, k)

### Optimisation

PEN_STR = "logL1"; # one of < L1 | logL1 | approxL0 | hslike >, see end of `src/inference.jl` for details about penalty functions

# Default hyperparameter values
HYP_DICT = Dict(
	"L1" => 20.0, # manually chosen
	"logL1" => 1.0, # taken from Gupta et al. (2020)
	"approxL0" => log(length(data)), # BIC-inspired
	"hslike" => 20.0 # manually chosen
);

hyp_val = HYP_DICT[PEN_STR];
OPT_DIR = joinpath(@__DIR__, "output", PEN_STR, "hyp_$(hyp_val)") # directory for storing optimisation results
mkpath(OPT_DIR); # create directory 
N_RUNS = 16; # number of optimisation runs

# Create ODEInferenceProb struct, see `src/inference.jl` for documentation of fields
iprob = make_iprob(oprob, k, t_obs, data, σs_init, PEN_STR, hyp_val; scale_fcts);

# `iprob.tf` accounts for log transform and rate scaling
true_θ = [σs_true; iprob.tf(true_kvec)];
true_penloss = iprob.optim_func(true_θ)
iprob.loss_func(true_kvec, fill(0.01, 3))

## Test type stability and computational time
# @code_warntype iprob.optim_func([fill(0.01, 3); iprob.tf(true_kvec)])

# using BenchmarkTools
# @btime iprob.optim_func(true_θ)

# using ForwardDiff
# cfg = ForwardDiff.GradientConfig(iprob.optim_func, true_θ);
# @btime ForwardDiff.gradient(iprob.optim_func, $true_θ, $cfg)


# Random initial points for optimisation runs
# using Distributions
rng = StableRNG(1);
init_vec = [rand(rng, n_rx) for _ in 1:N_RUNS];
# init_vec = [rand(rng, Beta(1, n_rx), n_rx) for _ in 1:100];

# Function for displaying progress of each run, takes in run index (integer) and optimisation result as input
function optim_callback(i, res)
    time_str = pyfmt(FMT_2DP, res.time_run)
    # loss offset = diff b/w the local minimum found and the loss evaluated at the ground truth rate constants
    # omit subtraction if ground truth is not available
    penloss_diff_str = pyfmt(FMT_2DP, res.minimum - true_penloss)    
    println("Run $i: $(time_str) seconds, penalised loss diff. = $(penloss_diff_str)")
    flush(stdout)
end

# Perform multi-start optimisation
@time res_vec = optim_iprob(iprob, lbs, ubs, init_vec; callback_func=optim_callback);

est_σs = res_vec[2].minimizer[1:3]
est_kvec = iprob.itf(res_vec[2].minimizer[4:end])
iprob.loss_func(est_kvec, est_σs) - n_obs * sum(log, est_σs) # should be N/2

# Export estimated reaction rates
export_estimates(res_vec, OPT_DIR);


# Read in reactions rates
est_mat = readdlm(joinpath(OPT_DIR, "estimates.txt"));
# Visualise results
make_plots_runs(iprob, est_mat, true_θ, k, OPT_DIR);


### Evaluation
true_rx = findall(>(0.0), true_kvec);
optim_vals = iprob.optim_func.(eachcol(est_mat)); # optimised loss value for each run
est = est_mat[:,argmin(optim_vals)];
est_kvec = iprob.itf(est[n_species+1:end]) # rate constants for best run

# sum of errors for truly present reactions
sum(abs.(est_kvec[true_rx] .- true_kvec[true_rx]))
# sum of all other reaction rates
sum(est_kvec) - sum(est_kvec[true_rx])

infer_reactions(make_isol(iprob, est_mat))
