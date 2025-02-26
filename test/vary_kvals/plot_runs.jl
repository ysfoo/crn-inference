#################################################################################
### Plots results (all runs) for different datasets and optimisation settings ###
#################################################################################

include(joinpath(@__DIR__, "setup.jl")); # set up optimisation problem
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));

σs_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
scale_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
for k1 in K1_VALS, k18 in K18_VALS
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	smoothers = smooth_data.(eachrow(data), Ref(t_obs));
	σs_dict[(k1, k18)] = estim_σ.(eachrow(data), smoothers);
	scale_dict[(k1, k18)] = get_scale_fcts(smoothers, species_vec, rx_vec, k);
end

# Visualise results (no multi-threading as plotting is not thread-safe, boo!)
for (k1, k18, pen_str, hyp_val) in settings_vec
    true_kvec = make_true_kvec(k1, k18);

    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));

    σs_init = σs_dict[(k1, k18)]
	scale_fcts = scale_dict[(k1, k18)]

	opt_dir = get_opt_dir(k1, k18, pen_str, hyp_val)

    iprob = make_iprob(oprob, k, t_obs, data, σs_init, pen_str, hyp_val; scale_fcts);
	est_mat = readdlm(joinpath(opt_dir, "estimates.txt"));
	make_plots_runs(iprob, est_mat, true_kvec, σs_true, k, opt_dir);
    GC.gc(); # garbage collection, just in case
end
