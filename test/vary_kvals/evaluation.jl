#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));

t_grid = range(t_span..., 1000);

scale_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
for k1 in K1_VALS, k18 in K18_VALS
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
	scale_dict[(k1, k18)] = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)

    # display((k1,k18))
    # init_σs = [
	# 	estim_σ(datarow .- eval_spline(res..., t_obs)) 
	# 	for (datarow, res) in zip(eachrow(data), smooth_resvec)
	# ]
    # display(init_σs ./ vec(readdlm(joinpath(data_dir, "stds_true.txt"))))
    
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

for k1 in K1_VALS, k18 in K18_VALS, pen_str in PEN_STRS
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

    tune_idx = tune_hyp_bic(inferred_vec, fit_vec)
    hyp_choice_dict[(k1, k18, pen_str)] = HYP_VALS[tune_idx]

    # continue
    # Plotting
    ns = length.(inferred_vec)
    nmin, nmax = extrema(ns)
    f = Figure()
    ax1 = Axis(
        f[1, 1], yticklabelcolor=:dodgerblue4, xscale=log10, 
        xlabel=L"\lambda", ylabel="Model fit heuristic",
        title=L"$%$(pen_names[pen_str])$ \textbf{penalty}, $k_1 = %$(k1)$, $k_{18} = %$(k18)$"
    )
    ax2 = Axis(
        f[1, 1], yticklabelcolor=:darkorange3, yticks=nmin:nmax,
        xscale=log10, ylabel="Number of reactions inferred", yaxisposition = :right
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



# Directory for evaluation plots
eval_dir = joinpath(@__DIR__, "output/eval_figs");
mkpath(eval_dir);

## Network reconstruction

true_rxs = [1, 18, 13];
tpos_dict = Dict(settings => length(rxs ∩ true_rxs) for (settings, rxs) in inferred_dict); # true positives
fpos_dict = Dict(settings => length(setdiff(rxs, true_rxs)) for (settings, rxs) in inferred_dict); # false positives
fneg_dict = Dict(settings => length(setdiff(true_rxs, rxs)) for (settings, rxs) in inferred_dict); # false negatives

begin
fig = Figure(size=(1000, 600));
# True positives
for ax_j in eachindex(PEN_STRS)
    ax = Axis(
        fig[1, ax_j], aspect=DataAspect(),
        xminorgridcolor = :black,
        xminorgridvisible = true,
        xminorticks = eachindex(K1_VALS) .- 0.5,
        yminorgridcolor = :black,
        yminorgridvisible = true,
        yminorticks = eachindex(K1_VALS) .- 0.5,
        xticks = (eachindex(K1_VALS), string.(K1_VALS)),
        yticks = (eachindex(K18_VALS), string.(K18_VALS)),
        xlabel = L"Ground truth $k_1$",
        ylabel = L"Ground truth $k_{18}$",
        title = L"$%$(pen_names[PEN_STRS[ax_j]])$ \textbf{penalty}"
    )
    
    mat = [
        begin
            settings = (k1, k18, PEN_STRS[ax_j]);
            tpos_dict[(settings..., hyp_choice_dict[settings])]
        end for k1 in K1_VALS, k18 in K18_VALS]
    hm = heatmap!(mat, colormap=(:greens, 0.8), colorrange=(0, 3))
    for i in eachindex(K1_VALS), j in eachindex(K18_VALS)
        text!(ax, "$(mat[i, j])", position = (i, j), fontsize=16,
            color = :black, align = (:center, :center))
    end
end
Label(
    fig[0, 1:length(PEN_STRS)], "Number of reactions correctly identified", 
    font = :bold, fontsize=20
);
Label(
    fig[2, 1:length(PEN_STRS)], "Number of reactions spuriously identified", 
    font = :bold, fontsize=20
);
# False positives
fpos_max = maximum(
    begin
        settings = (k1, k18, pen_str);
        fpos_dict[(settings..., hyp_choice_dict[settings])]
    end for k1 in K1_VALS, k18 in K18_VALS, pen_str in PEN_STRS)
for ax_j in eachindex(PEN_STRS)
    ax = Axis(
        fig[3, ax_j], aspect=DataAspect(),
        xminorgridcolor = :black,
        xminorgridvisible = true,
        xminorticks = eachindex(K1_VALS) .- 0.5,
        yminorgridcolor = :black,
        yminorgridvisible = true,
        yminorticks = eachindex(K1_VALS) .- 0.5,
        xticks = (eachindex(K1_VALS), string.(K1_VALS)),
        yticks = (eachindex(K18_VALS), string.(K18_VALS)),
        xlabel = L"Ground truth $k_1$",
        ylabel = L"Ground truth $k_{18}$",
        title = L"$%$(pen_names[PEN_STRS[ax_j]])$ \textbf{penalty}"
    )
    mat = [
        begin
            settings = (k1, k18, PEN_STRS[ax_j]);
            fpos_dict[(settings..., hyp_choice_dict[settings])]
        end for k1 in K1_VALS, k18 in K18_VALS]
    hm = heatmap!(mat, colormap=(:reds, 0.8), colorrange=(0, fpos_max), colorscale=sqrt)
    for i in eachindex(K1_VALS), j in eachindex(K18_VALS)
        text!(ax, "$(mat[i, j])", position = (i, j),
            color = :black, align = (:center, :center))
    end
end
save(joinpath(eval_dir, "eval_network.png"), fig);
end

## Rate constant estimation

function make_kerr_plot(k_idx, true_k_dict, fig, ax_i)
    k_err_dict = Dict(
        settings => begin
            true_k = true_k_dict[settings];
            est_kvec = mask_kvec(isol_dict[settings], inferred_dict[settings])
            (est_kvec[k_idx] - true_k) / true_k
        end for (settings, isol) in isol_dict
    );
    for (ax_j, pen_str) in enumerate(PEN_STRS)
        ax = Axis(
            fig[ax_i, ax_j], aspect=DataAspect(),
            xminorgridcolor = :black,
            xminorgridvisible = true,
            xminorticks = eachindex(K1_VALS) .- 0.5,
            yminorgridcolor = :black,
            yminorgridvisible = true,
            yminorticks = eachindex(K1_VALS) .- 0.5,
            xticks = (eachindex(K1_VALS), string.(K1_VALS)),
            yticks = (eachindex(K18_VALS), string.(K18_VALS)),
            xlabel = L"Ground truth $k_1$",
            ylabel = L"Ground truth $k_{18}$",
            title = L"%$(pen_names[PEN_STRS[ax_j]])"
        )
        mat = [
            begin
                hyp_val = hyp_choice_dict[(k1, k18, pen_str)];
                k_err_dict[(k1, k18, pen_str, hyp_val)]
            end for k1 in K1_VALS, k18 in K18_VALS]
        hm = heatmap!(mat, colormap=(:curl, 0.9), colorrange=(-1, 1))
    end
    Label(
        fig[ax_i-1, 1:length(PEN_STRS)], L"$\mathbf{k_{%$(k_idx)}} \textbf{ error (relative)}$", 
        fontsize=20,
    )
end

true_k1_dict = Dict(settings => settings[1] for settings in settings_vec);
true_k13_dict = Dict(settings => 1.0 for settings in settings_vec);
true_k18_dict = Dict(settings => settings[2] for settings in settings_vec);

begin
fig = Figure(size=(1000, 800))
make_kerr_plot(1, true_k1_dict, fig, 1);
make_kerr_plot(13, true_k13_dict, fig, 3);
make_kerr_plot(18, true_k18_dict, fig, 5);
Colorbar(fig[:, end+1], limits=(-1, 1), colormap=:curl, height = Relative(0.9), ticklabelsize=18)
save(joinpath(eval_dir, "k_err.png"), fig);
end


## Trajectory reconstruction and prediction

x0_altmap = [:X1 => 1., :X2 => 0., :X3 => 0.];
traj_err_dict = Dict{Tuple, AbstractMatrix}();
pred_err_dict = Dict{Tuple, AbstractMatrix}();

for k1 in K1_VALS, k18 in K18_VALS, pen_str in PEN_STRS
    hyp_val = hyp_choice_dict[(k1, k18, pen_str)]
    settings = (k1, k18, pen_str, hyp_val)
    true_kvec = make_true_kvec(k1, k18);
    isol = isol_dict[settings]      
    est_kvec = mask_kvec(isol, inferred_dict[settings]) # zero out non-inferred reactions
    traj_err_dict[settings] = get_traj_err(
        est_kvec, true_kvec, isol.iprob.oprob, k, t_grid
    )
    pred_err_dict[settings] = get_traj_err(
        est_kvec, true_kvec, isol.iprob.oprob, k, t_grid; u0=x0_altmap
    )
end

grid_diff = diff(t_grid);
traj_L1_dict = Dict(
    settings => get_traj_L1(traj_err, grid_diff) 
    for (settings, traj_err) in traj_err_dict
);
traj_L1_min, traj_L1_max = extrema(values(traj_L1_dict))
pred_L1_dict = Dict(
    settings => get_traj_L1(pred_err, grid_diff) 
    for (settings, pred_err) in pred_err_dict
);
pred_L1_min, pred_L1_max = extrema(values(pred_L1_dict))
colorrange = (min(traj_L1_min, pred_L1_min), max(traj_L1_max, pred_L1_max))

begin
    fig = Figure(size=(1000, 600))
    for (ax_i, pen_str) in enumerate(PEN_STRS)
        ax = Axis(
            fig[1, ax_i], aspect=DataAspect(),
            xminorgridcolor = :black,
            xminorgridvisible = true,
            xminorticks = eachindex(K1_VALS) .- 0.5,
            yminorgridcolor = :black,
            yminorgridvisible = true,
            yminorticks = eachindex(K1_VALS) .- 0.5,
            xticks = (eachindex(K1_VALS), string.(K1_VALS)),
            yticks = (eachindex(K18_VALS), string.(K18_VALS)),
            xlabel = L"Ground truth $k_1$",
            ylabel = L"Ground truth $k_{18}$",
            title = L"%$(pen_names[PEN_STRS[ax_i]])"
        )
        
        mat = [
            begin
                hyp_val = hyp_choice_dict[(k1, k18, pen_str)];
                traj_L1_dict[(k1, k18, pen_str, hyp_val)]
            end for k1 in K1_VALS, k18 in K18_VALS]
            hm = heatmap!(mat, colorscale=log10, colormap=(:Reds, 0.8), colorrange=colorrange)
    end;
    Label(
        fig[0, 1:length(PEN_STRS)], "Trajectory reconstruction error (absolute)", 
        font = :bold, fontsize=20
    );
    for (ax_i, pen_str) in enumerate(PEN_STRS)
        ax = Axis(
            fig[3, ax_i], aspect=DataAspect(),
            xminorgridcolor = :black,
            xminorgridvisible = true,
            xminorticks = eachindex(K1_VALS) .- 0.5,
            yminorgridcolor = :black,
            yminorgridvisible = true,
            yminorticks = eachindex(K1_VALS) .- 0.5,
            xticks = (eachindex(K1_VALS), string.(K1_VALS)),
            yticks = (eachindex(K18_VALS), string.(K18_VALS)),
            xlabel = L"Ground truth $k_1$",
            ylabel = L"Ground truth $k_{18}$",
            title = L"%$(pen_names[PEN_STRS[ax_i]])"
        )
        
        mat = [
            begin
                hyp_val = hyp_choice_dict[(k1, k18, pen_str)];
                pred_L1_dict[(k1, k18, pen_str, hyp_val)]
            end for k1 in K1_VALS, k18 in K18_VALS]
        hm = heatmap!(mat, colorscale=log10, colormap=(:Reds, 0.8), colorrange=colorrange)
        if ax_i == length(PEN_STRS)
            Colorbar(fig[:, end+1], hm, height = Relative(0.9), ticklabelsize=18)
        end
    end;
    Label(
        fig[2, 1:length(PEN_STRS)], "Trajectory prediction error (absolute)", 
        font = :bold, fontsize=20
    );
    save(joinpath(eval_dir, "traj_err.png"), fig);
end

exit()

### Playground

k1 = 10.; k18 = .3; pen_str = "logL1";

[hyp_val => inferred_dict[(k1, k18, pen_str, hyp_val)] for hyp_val in HYP_VALS]
[hyp_val => fit_dict[(k1, k18, pen_str, hyp_val)] for hyp_val in HYP_VALS]
hyp_choice_dict[(k1, k18, pen_str)]

inferred_vec = [inferred_dict[(k1, k18, pen_str, hyp_val)] for hyp_val in HYP_VALS];
fit_vec = [fit_dict[(k1, k18, pen_str, hyp_val)] for hyp_val in HYP_VALS];
tune_idx = tune_hyp_bic(inferred_vec, fit_vec)

tune_hyp_plateau(inferred_vec, fit_vec)

settings = (k1, k18, pen_str, .125);
# inferred_dict[settings];
isol = isol_dict[settings];
isol.kvec
isol.σs

infer_reactions(isol, species_vec, rx_vec, print_diff=true)

true_kvec = make_true_kvec(k1, k18);
est_kvec = mask_kvec(isol, inferred_vec[tune_idx])

# reconstruction
begin
s_idx = 3;
# scatter(isol.iprob.t_obs, isol.iprob.data[s_idx,:]);
oprob = remake(isol.iprob.oprob; p=[isol.iprob.k => true_kvec]);
sol = solve(oprob, AutoTsit5(Rosenbrock23()); saveat=t_grid);
lines(t_grid, sol[s_idx,:]);
oprob = remake(isol.iprob.oprob; p=[isol.iprob.k => est_kvec]);
sol = solve(oprob, AutoTsit5(Rosenbrock23()); saveat=t_grid);
lines!(t_grid, sol[s_idx,:]);
display(current_figure())
end

# prediction
begin
    s_idx = 3;
    oprob = remake(isol.iprob.oprob; p=[isol.iprob.k => true_kvec], u0=x0_altmap);
    sol = solve(oprob, AutoTsit5(Rosenbrock23()); saveat=t_grid);
    lines(t_grid, sol[s_idx,:]);
    oprob = remake(isol.iprob.oprob; p=[isol.iprob.k => est_kvec], u0=x0_altmap);
    sol = solve(oprob, AutoTsit5(Rosenbrock23()); saveat=t_grid);
    lines!(t_grid, sol[s_idx,:]);
    display(current_figure())
    end


hyp_val = .25
settings = (k1, k18, pen_str, hyp_val);
true_kvec = make_true_kvec(k1, k18);
isol = isol_dict[settings];
est_kvec = mask_kvec(isol, inferred_dict[settings]) # zero out non-inferred reactions
maximum.(abs, eachrow(get_traj_err(
    est_kvec, true_kvec, isol.iprob.oprob, k, t_grid
)))

oprob = remake(isol.iprob.oprob; p=[isol.iprob.k => true_kvec]);
sol = solve(oprob, AutoTsit5(Rosenbrock23()); saveat=isol.iprob.t_obs);

scatter(isol.iprob.t_obs, isol.iprob.data[1,:])
lines!(isol.iprob.t_obs, sol[1,:]);

isol.iprob.loss_func(isol.kvec,isol.σs)
# isol.iprob.optim_func(isol.est)

nothing

# infer_reactions(isol, species_vec, rx_vec; print_diff=true)
# est_kvec = mask_kvec(isol, inferred_dict[settings])

# refine true_θ?
# data_dir = get_data_dir(k1, k18);
# true_σs = vec(readdlm(joinpath(data_dir, "stds_true.txt")))
# true_kvec = make_true_kvec(k1, k18);
# true_θ = [true_σs; isol.iprob.tf(true_kvec)];
# isol.iprob.optim_func(true_θ)
# res = optim_iprob(isol.iprob, fill(LB, n_rx), fill(UB, n_rx), [true_kvec ./ isol.iprob.scale_fcts])[1];
# isol.iprob.loss_func(isol.iprob.itf(res.minimizer[4:end]),res.minimizer[1:3])
# res.minimum


### Guesses for σ
# for k1 in K1_VALS, k18 in K18_VALS
#     true_kvec = make_true_kvec(k1, k18);
#     data_dir = get_data_dir(k1, k18);
#     t_obs, data = read_data(joinpath(data_dir, "data.txt"));
#     smoothers = smooth_data.(eachrow(data), Ref(t_obs); n_itp=50, d=2);
#     σs = estim_σ.(eachrow(data), smoothers)
#     println(k1, " ", k18)
#     display(σs)
# end