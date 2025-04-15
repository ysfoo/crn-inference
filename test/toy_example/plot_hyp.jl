#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

using ProgressMeter

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));

t_grid = range(t_span..., 1000);
# t_itp = range(0, sqrt(t_span[2]), 50) .^ 2;

scale_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
for k1 in K1_VALS, k18 in K18_VALS
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
	scale_dict[(k1, k18)] = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)

    # display((k1,k18))
    init_σs = [
		estim_σ(datarow .- eval_spline(res..., t_obs)) 
		for (datarow, res) in zip(eachrow(data), smooth_resvec)
	]
    true_σs = vec(readdlm(joinpath(data_dir, "stds_true.txt")))
    # display(init_σs./true_σs)
    # display([init_σs./true_σs std.(eachrow(data))./true_σs])
    
    # plot smoother
	# f = Figure()
    # ax = Axis(f[1,1], xlabel=L"t", ylabel="Concentration")
    # for i in eachindex(smooth_resvec)
    #     lines!(t_grid, eval_spline(smooth_resvec[i]..., t_grid))
    # end
    # for i in axes(data, 1)
    #     scatter!(t_obs, data[i,:])
    # end
    # display(current_figure())
end

# Text to appear on plots
pen_names = Dict(
    "L1" => "\\mathbf{L_1}", 
    "logL1" => "\\textbf{log } \\mathbf{L_1}", 
    "approxL0" => "\\textbf{Approximate }\\mathbf{L_0}", 
    "hslike" => "\\textbf{Horseshoe}"
);

# Directory for hyperparameter tuning plots
tune_dir = joinpath(@__DIR__, "output/vary_hyp_figs");
for pen_str in PEN_STRS
    mkpath(joinpath(tune_dir, pen_str))
end

isol_dict = Dict{Tuple, ODEInferenceSol}();
hyp_choice_dict = Dict{Tuple, Float64}();
inferred_dict = Dict{Tuple, Vector{Int64}}();
fit_dict = Dict{Tuple, Float64}();

@showprogress for (k1, k18, pen_str) in Iterators.product(K1_VALS, K18_VALS, PEN_STRS)
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
    
	scale_fcts = scale_dict[(k1, k18)]

    opt_dirs = get_opt_dir.(k1, k18, Ref(pen_str), HYP_VALS)
    iprobs = [
        make_iprob(oprob, k, t_obs, data, pen_str, hyp_val; scale_fcts)
        for hyp_val in HYP_VALS]
    isol_vec = [
            begin
                est_mat = readdlm(joinpath(opt_dir, "estimates.txt"));
                make_isol(iprob, est_mat)
            end for (opt_dir, iprob) in zip(opt_dirs, iprobs)]

    for (hyp_val, isol) in zip(HYP_VALS, isol_vec)
        settings = (k1, k18, pen_str, hyp_val)        
        inferred, fit = infer_reactions(isol, species_vec, rx_vec)
        isol_dict[settings] = isol
        inferred_dict[settings] = inferred
        fit_dict[settings] = fit
    end

    inferred_vec = [inferred_dict[(k1, k18, pen_str, hyp_val)] for hyp_val in HYP_VALS]
    fit_vec = [fit_dict[(k1, k18, pen_str, hyp_val)] for hyp_val in HYP_VALS]

    tune_idx = tune_hyp_bic(inferred_vec, fit_vec, length(data))
    hyp_choice_dict[(k1, k18, pen_str)] = HYP_VALS[tune_idx]

    # continue
    # Plotting
    ns = length.(inferred_vec)
    nmin, nmax = extrema(ns)
    f = Figure()
    ax1 = Axis(
        f[1, 1], yticklabelcolor=:dodgerblue4,
        xlabel=L"\lambda", ylabel="Model fit heuristic",
        title=L"$%$(pen_names[pen_str])$ \textbf{penalty}, $k_1 = %$(k1)$, $k_{18} = %$(k18)$"
    )
    ax2 = Axis(
        f[1, 1], yticklabelcolor=:darkorange3, yticks=nmin:nmax,
        ylabel="Number of reactions inferred", yaxisposition = :right
    )
    hidespines!(ax2)
    hidexdecorations!(ax2)
    scatterlines!(
        ax1, HYP_VALS, fit_vec
    )
    scatterlines!(
        ax2, HYP_VALS, ns, 
        color=palette[2]
    )
    scatter!(ax1, HYP_VALS[tune_idx], fit_vec[tune_idx]; color=:dodgerblue4, marker=:star5, markersize=20)
    scatter!(ax2, HYP_VALS[tune_idx], ns[tune_idx]; color=:darkorange4, marker=:star5, markersize=20)

    k1_str = pyfmt(".0e", k1)
    k18_str = pyfmt(".0e", k18)
    save(joinpath(tune_dir, pen_str, "k1_$(k1_str)_k18_$(k18_str).png"), f)
    # display(current_figure())
end

exit()
