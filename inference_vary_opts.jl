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

# Define ground truth
true_kvec = zeros(n_rx);
true_kvec[1] = true_kvec[18] = true_kvec[13] = 1.;

# Import synthetic data
t_obs, data = read_data(joinpath(@__DIR__, "output"));
n_obs = length(t_obs);

# Optimisation setup
N_RUNS = 15; # number of optimisation runs
init_vec = rand_inits(N_RUNS, n_rx, 1); # Vector of random initial points for optimisation runs

# All possible optimisation options
opt_options = Iterators.product(["L1", "logL1", "approxL0", "hslike"], [true, false]);

HYP_DICT = Dict(
	"L1" => 20.0, 
	"logL1" => 1.0, 
	"approxL0" => log(length(data)), 
	"hslike" => 20.0
);

# Perform multi-start optimisation and export reaction rates
# Comment out this block if optimisation is already performed (e.g. when redoing plots)
Threads.@threads for (PEN_STR, LOG_OPT) in collect(opt_options)
	OPT_DIRNAME = joinpath(@__DIR__, "output/vary_opts", PEN_STR * "_" * (LOG_OPT ? "uselog" : "nolog")) # directory for storing results
	mkpath(OPT_DIRNAME);

	pen_hyp = HYP_DICT[PEN_STR]
	iprob = make_iprob(oprob, t_obs, data, PEN_STR, LB, k, pen_hyp; log_opt=LOG_OPT)
	true_loss = iprob.optim_func(iprob.tf.(true_kvec))

	open(joinpath(OPT_DIRNAME, "optim_progress.txt"), "w") do io
		function optim_callback(i, res)
			time_str = pyfmt(FMT_2DP, res.time_run)
			# loss offset = diff b/w the local minimum found and the loss evaluated at the ground truth rate constants
			loss_offset_str = pyfmt(FMT_2DP, res.minimum - true_loss)    
			write(io, "Run $i: $(time_str) seconds, loss offset = $(loss_offset_str)\n")
			flush(io)
		end
		res_vec = optim_iprob(iprob, lbs, ubs, init_vec; callback_func=optim_callback)
		export_estimates(res_vec, iprob, OPT_DIRNAME, "inferred_rates.txt");
	end
end

# Visualise results (no multi-threading as plotting is not thread-safe, boo!)
for (PEN_STR, LOG_OPT) in collect(opt_options)
	OPT_DIRNAME = joinpath(@__DIR__, "output/vary_opts", PEN_STR * "_" * (LOG_OPT ? "uselog" : "nolog")) # directory for storing results

	pen_hyp = HYP_DICT[PEN_STR]
	iprob = make_iprob(oprob, t_obs, data, PEN_STR, LB, k, pen_hyp; log_opt=LOG_OPT)

	kmat = readdlm(joinpath(OPT_DIRNAME, "inferred_rates.txt"));
	make_plots(iprob, kmat, true_kvec, k, OPT_DIRNAME);
end
