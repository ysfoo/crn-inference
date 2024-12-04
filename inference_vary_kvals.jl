## Performs inference for different optimisation options
## Ground truth rate constants are all 1.0

using Format
FMT_2DP = ".2f"; # `pyfmt(FMT_2DP, num)` converts a float `num` to a string with 2 decimal points

include(joinpath(@__DIR__, "full_network.jl")); # defines `full_network` and `k` (Symbolics object for rate constants)
include(joinpath(@__DIR__, "inference.jl")); # imports key functions

# Setup
σ = 0.01; # assume noise SD is known
x0map = [:X1 => 0., :X2 => 0., :X3 => 1.]; # assume initial conditions are known
t_span = (0., 10.);
rx_vec = Catalyst.reactions(full_network); # list of reactions
n_rx = length(rx_vec);
oprob = ODEProblem(full_network, x0map, t_span, zeros(n_rx)); # all rates here are zero, no dynamics

LB, UB = 1e-10, 1e2; # bounds for reaction rate constants (i.e. parameters)
lbs = LB .* ones(n_rx);
ubs = UB .* ones(n_rx);

# Optimisation setup
N_RUNS = 15; # number of optimisation runs
init_vec = rand_inits(N_RUNS, n_rx, 1); # Vector of random initial points for optimisation runs

# All possible setups
k1_choice = [0.1, 0.3, 1., 3., 10.];
k18_choice = [0.1, 0.3, 1., 3., 10.];
pen_choice = ["L1", "logL1", "approxL0", "hslike"]; # penalty_function
hyp_choice = ["half_hyps", "orig_hyps", "double_hyps"] # hyperparameter values (relative to default)
setups = collect(Iterators.product(
    k1_choice, k18_choice, pen_choice, hyp_choice
));

function make_data_dir(k1, k18)
    k1_str = pyfmt(".0e", k1)
    k18_str = pyfmt(".0e", k18)
    return joinpath(@__DIR__, "output/vary_kvals/k1_$(k1_str)_k18_$(k18_str)")
end

HYP_DICT = Dict(
    "L1" => 20.0, 
    "logL1" => 1.0, 
    "approxL0" => log(303), # total number of data points 
    "hslike" => 20.0
);

MULT_DICT = Dict("orig_hyps" => 1.0, "half_hyps" => 0.5, "double_hyps" => 2.0);

# Perform multi-start optimisation and export reaction rates
# Comment out this block if optimisation is already performed (e.g. when redoing plots)
Threads.@threads for (k1, k18, pen_str, hyp_str) in collect(setups)
    # Define ground truth
    true_kvec = zeros(n_rx);
    true_kvec[1] = k1; true_kvec[18] = k18; true_kvec[13] = 1.;

    # Import synthetic data    
    data_dir = make_data_dir(k1, k18)
    t_obs, data = read_data(data_dir);
    n_obs = length(t_obs);
	
    # Make directory
	opt_dirname = joinpath(data_dir, hyp_str, pen_str * "_uselog") # directory for storing results
	mkpath(opt_dirname);

    # Optimisation
    pen_hyp = HYP_DICT[pen_str] * MULT_DICT[hyp_str]
	iprob = make_iprob(oprob, t_obs, data, pen_str, LB, k, pen_hyp)
	true_loss = iprob.optim_func(iprob.tf.(true_kvec))

	open(joinpath(opt_dirname, "optim_progress.txt"), "w") do io
		function optim_callback(i, res)
			time_str = pyfmt(FMT_2DP, res.time_run)
			# loss offset = diff b/w the local minimum found and the loss evaluated at the ground truth rate constants
			loss_offset_str = pyfmt(FMT_2DP, res.minimum - true_loss)    
			write(io, "Run $i: $(time_str) seconds, loss offset = $(loss_offset_str)\n")
			flush(io)
		end
		res_vec = optim_iprob(iprob, lbs, ubs, init_vec; callback_func=optim_callback)
		export_estimates(res_vec, iprob, opt_dirname, "inferred_rates.txt");
	end
end

# Visualise results (no multi-threading as plotting is not thread-safe, boo!)
for (k1, k18, pen_str, hyp_str) in collect(setups)
    true_kvec = zeros(n_rx);
    true_kvec[1] = k1; true_kvec[18] = k18; true_kvec[13] = 1.;

    data_dir = make_data_dir(k1, k18)
    t_obs, data = read_data(data_dir);
    n_obs = length(t_obs);
    
	opt_dirname = joinpath(data_dir, hyp_str, pen_str * "_uselog") # directory for storing results
    pen_hyp = HYP_DICT[pen_str] * MULT_DICT[hyp_str]
	iprob = make_iprob(oprob, t_obs, data, pen_str, LB, k, pen_hyp)
    
	kmat = readdlm(joinpath(opt_dirname, "inferred_rates.txt"));
	make_plots(iprob, kmat, true_kvec, k, opt_dirname);
end
