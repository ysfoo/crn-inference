############################################
### Perform inference for a single setup ###
############################################

using Format
FMT_2DP = ".2f" # `pyfmt(FMT_2DP, num)` converts a float `num` to a string with 2 decimal points

include(joinpath(@__DIR__, "full_network.jl")); # defines `full_network` and `k` (Symbolics object for rate constants)
include(joinpath(@__DIR__, "inference.jl")); # imports key functions
# Key functions imported: read_data, make_iprob, rand_inits, optim_iprob, export_estimates, make_plots

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
t_obs, data = read_data(joinpath(@__DIR__, "output"));
n_obs = length(t_obs);

# Verify full CRN can reproduce 'ground truth' CRN
true_kvec = zeros(n_rx);
true_kvec[1] = true_kvec[18] = true_kvec[13] = 1.;
sol = solve(remake(oprob, p=[k => true_kvec]));

sol.u[end] # final point for trajectory simulated from full model
data[:,end] # final data point (noisy), should be similar to above


### Optimisation setup
PEN_STR = "L1"; # one of < L1 | logL1 | approxL0 | hslike >, see end of `inference.jl` for details about penalty functions
LOG_OPT = true; # whether to perform optimisation in log space of parameters
PEN_HYP = nothing; # leave as nothing to use default hyperparameter
if PEN_STR == "approxL0"
    PEN_HYP = log(n_obs) / 2; # BIC-inspired hyparameter for the approxL0 penalty function
end
OPT_DIRNAME = joinpath(@__DIR__, "output/vary_opts", PEN_STR * "_" * (LOG_OPT ? "uselog" : "nolog")) # directory for storing optimisation results
mkpath(OPT_DIRNAME);
N_RUNS = 15; # number of optimisation runs

init_vec = rand_inits(N_RUNS, n_rx, 2024); # Vector of random initial points for optimisation runs

# Create ODEInferenceProb struct, see `inference.jl` for documentation of fields
iprob = make_iprob(oprob, t_obs, data, PEN_STR, LB, k; log_opt=LOG_OPT, pen_hyp=PEN_HYP);

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
    loss_offset_str = pyfmt(FMT_2DP, res.minimum - true_loss)    
    println("Run $i: $(time_str) seconds, loss offset = $(loss_offset_str)")
    flush(stdout)
end


### Perform multi-start optimisation
out_file = open(joinpath(OPT_DIRNAME, "optim_progress.txt"), "w"); # file for outputting optimisation progress
res_vec = redirect_stdout(out_file) do
    optim_iprob(iprob, lbs, ubs, init_vec; callback_func=optim_callback)
end;
close(out_file)

# Export estimated reaction rates
export_estimates(res_vec, iprob, OPT_DIRNAME, "inferred_rates.txt");


### Visualise results
kmat = readdlm(joinpath(OPT_DIRNAME, "inferred_rates.txt")); # read in reactions rates
make_plots(iprob, kmat, true_kvec, k, OPT_DIRNAME);