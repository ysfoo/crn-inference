###################################################################
### Perform inference for a single instance and produce results ###
###################################################################

using DelimitedFiles
using Random

using Format
FMT_2DP = ".2f"; # `pyfmt(FMT_2DP, num)` converts a float `num` to a string with 2 decimal points

include(joinpath(@__DIR__, "../define_networks.jl")); # defines `full_network` and `k` (Symbolics object for rate constants)
include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference
include(joinpath(@__DIR__, "../evaluation.jl")); # imports functions used for results

### Setup
σ = 0.01; # assume noise SD is known
x0map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # assume initial conditions are known
t_span = (0., 10.);
rx_vec = Catalyst.reactions(full_network); # list of reactions
n_rx = length(rx_vec);
oprob = ODEProblem(full_network, x0map, t_span, zeros(n_rx)); # all rates here are zero, no dynamics

LB, UB = 1e-10, 1e2; # bounds for reaction rate constants (i.e. parameters)
lbs = LB .* ones(n_rx);
ubs = UB .* ones(n_rx);


### Import synthetic data
data_fname = joinpath(@__DIR__, "output", "data.txt")
fullmat = readdlm(data_fname);
t_obs = fullmat[:,1];
data = fullmat[:,2:end]';
n_obs = length(t_obs);

# Verify full CRN can reproduce 'ground truth' CRN
true_kvec = zeros(n_rx);
true_kvec[1] = true_kvec[18] = true_kvec[13] = 1.;
sol = solve(remake(oprob, p=[k => true_kvec]));

sol.u[end] # final point for trajectory simulated from full model
data[:,end] # final data point (noisy), should be similar to above


### Optimisation
PEN_STR = "L1"; # one of < L1 | logL1 | approxL0 | hslike >, see end of `src/inference.jl` for details about penalty functions
LOG_OPT = true; # whether to perform optimisation in log space of parameters

# Default hyperparameter values
HYP_DICT = Dict(
	"L1" => 20.0, # manually chosen
	"logL1" => 1.0, # taken from Gupta et al. (2020)
	"approxL0" => log(length(data)), # BIC-inspired
	"hslike" => 20.0 # manually chosen
);

pen_hyp = HYP_DICT[PEN_STR];
OPT_DIR = joinpath(@__DIR__, "output", PEN_STR * "_" * (LOG_OPT ? "uselog" : "nolog")) # directory for storing optimisation results
mkpath(OPT_DIR); # create directory 
N_RUNS = 15; # number of optimisation runs

# Random initial points for optimisation runs
Random.seed!(1);
init_vec = [rand(n_rx) for _ in 1:N_RUNS]

# Create ODEInferenceProb struct, see `src/inference.jl` for documentation of fields
iprob = make_iprob(oprob, t_obs, data, PEN_STR, LB, k, pen_hyp; log_opt=LOG_OPT);

# Large loss value when evaluated with incorrect rate constants (all 1s)
iprob.optim_func(iprob.tf.(ones(n_rx))) # `iprob.tf` accounts for a potential log transform when `LOG_OPT == true`
# Smaller loss value when evaluated with the ground truth rate constants
true_loss = iprob.optim_func(iprob.tf.(true_kvec))

# Penalty function evaluated with the ground truth rate constants
true_pen = iprob.penalty_func.(true_kvec) # NB: no `iprob.tf` as penalty function is defined on the original space

# Function for displaying progress of each run, takes in run index (integer) and optimisation result as input
function optim_callback(i, res)
    time_str = pyfmt(FMT_2DP, res.time_run)
    # loss offset = diff b/w the local minimum found and the loss evaluated at the ground truth rate constants
    # omit subtraction if ground truth is not available
    loss_offset_str = pyfmt(FMT_2DP, res.minimum - true_loss)    
    println("Run $i: $(time_str) seconds, loss offset = $(loss_offset_str)")
    flush(stdout)
end

# Perform multi-start optimisation
res_vec = optim_iprob(iprob, lbs, ubs, init_vec; callback_func=optim_callback);

# Export estimated reaction rates
export_estimates(res_vec, iprob, OPT_DIR, "inferred_rates.txt");


# Read in reactions rates
kmat = readdlm(joinpath(OPT_DIR, "inferred_rates.txt"));
# Visualise results
make_plots_runs(iprob, kmat, true_kvec, k, OPT_DIR);


### Report errors
true_rx = findall(>(0.0), true_kvec);
optim_loss = iprob.optim_func.(eachcol(iprob.tf.(kmat))); # optimised loss value for each run
est_kvec = kmat[:,argmin(optim_loss)]; # rate constants for best run

# sum of errors for truly present reactions
sum(abs.(est_kvec[true_rx] .- true_kvec[true_rx]))
# sum of all other reaction rates
sum(est_kvec) - sum(est_kvec[true_rx])
