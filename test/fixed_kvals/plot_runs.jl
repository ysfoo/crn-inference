####################################################################
### Plots results (all runs) for different optimisation settings ###
####################################################################

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));

# Import synthetic data
t_obs, data = read_data(joinpath(@__DIR__, "data.txt"));
n_species, n_obs = size(data);

smoothers = smooth_data.(eachrow(data), Ref(t_obs));
σs_init = estim_σ.(eachrow(data), smoothers);
scale_fcts = get_scale_fcts(smoothers, species_vec, rx_vec, k);

# Visualise results
for (pen_str, hyp_val) in opt_options
	opt_dir = get_opt_dir(pen_str, hyp_val) # directory for storing results
	
	iprob = make_iprob(oprob, k, t_obs, data, σs_init, pen_str, hyp_val; scale_fcts);
	est_mat = readdlm(joinpath(opt_dir, "estimates.txt"));
	make_plots_runs(iprob, est_mat, true_kvec, σs_true, k, opt_dir);
end
