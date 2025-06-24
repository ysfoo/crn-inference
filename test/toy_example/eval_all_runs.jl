#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

using ProgressMeter
using LogExpFunctions, SpecialFunctions, StatsBase

include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "../../src/inference.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));

# NO_CROSS = false
# EST_FNAME = NO_CROSS ? "nocross_estimates.txt" : "refined_estimates.txt"
# PLT_FNAME = NO_CROSS ? "top_crns_nocross.png" : "top_crns_crossover.png"

true_rxs = [1, 13, 18];
equiv_crns = [[1,11,14,18],[1,12,15,18],[1,13,16,19],[1,13,17,20]];
# [[true_rxs]; equiv_crns]
t_grid = range(t_span..., 1000);

scale_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
for k1 in K1_VALS, k18 in K18_VALS
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
	scale_dict[(k1, k18)] = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)
end

## Text to appear on plots
pen_names = Dict(
    "L1" => "\\mathbf{L_1}", 
    "logL1" => "\\textbf{log } \\mathbf{L_1}", 
    "approxL0" => "\\textbf{Approximate }\\mathbf{L_0}", 
    "hslike" => "\\textbf{Horseshoe}"
);

pen_names_reg = Dict(
    "L1" => "L_1",
    "logL1" => "\\text{log }L_1",
    "approxL0" => "\\text{Approx. }L_0",
    "hslike" => "\\text{Horseshoe}"
);

logprior(n) = -logabsbinomial(n_rx, n)[1] - log(n_rx+1)

bic_dict_lookup = Dict{Tuple, Dict{Vector{Int64},Float64}}();
post_dict_lookup = Dict{Tuple, Dict{Vector{Int64},Float64}}();
isol_dict_lookup = Dict{Tuple, Dict{Vector{Int64},ODEInferenceSol}}();
top_95_dict = Dict{Tuple, Vector{Vector{Int64}}}();
crn_mode_dict = Dict{Tuple, Vector{Int64}}();
base_crns_dict = Dict{Tuple, Set{Vector{Int64}}}()

@showprogress for (k1, k18, pen_str) in Iterators.product(K1_VALS, K18_VALS, PEN_STRS)
    settings = (k1, k18, pen_str);
    data_dir = get_data_dir(k1, k18);
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
    scale_fcts = scale_dict[(k1, k18)];
    iprob = make_iprob(
        oprob, k, t_obs, data, pen_str, HYP_VALS[1]; 
        scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
    )
    bic_by_rxs = Dict{Vector{Int64},Float64}();
    isol_by_rxs = Dict{Vector{Int64},ODEInferenceSol}();
    est_mat = readdlm(joinpath(data_dir, pen_str, "refined_estimates.txt"));
    # est_mat[findall(est_mat .≈ log(1e-6))] .= -Inf
    for est in eachcol(est_mat)
        rxs = findall(exp.(est[1:n_rx]) .> 1e-6 + eps(Float64))
        kvec = iprob.itf(est[1:n_rx])
        σs = exp.(est[n_rx+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)      
        bic = 2*iprob.loss_func(kvec, σs) + length(rxs)*log(length(data))
        if (get(bic_by_rxs, rxs, Inf) < bic) continue end
        bic_by_rxs[rxs] = bic
        isol_by_rxs[rxs] = isol
    end
    bic_dict_lookup[settings] = bic_by_rxs
    isol_dict_lookup[settings] = isol_by_rxs

    logps = Dict(rxs => -0.5*bic+logprior(length(rxs)) for (rxs, bic) in bic_dict_lookup[settings])
    logZ = logsumexp(collect(values(logps)))
    post_dict_lookup[settings] = posts = Dict(rxs => exp(logp-logZ) for (rxs, logp) in logps)

    cutoff_idx = findfirst((>)(0.95), cumsum(sort(collect(values(posts)), rev=true)))
    cutoff_logp = sort(collect(values(logps)), rev=true)[cutoff_idx]
    top_95_dict[settings] = [rxs for (rxs, logp) in logps if logp >= cutoff_logp]
    crn_mode_dict[settings] = argmax(x->posts[x], keys(posts))

    base_crns = Set{Vector{Int64}}();
    opt_dir = joinpath(@__DIR__, "output", pen_str)
    est_mat = readdlm(joinpath(data_dir, pen_str, "nocross_estimates.txt"))
    # est_mat[findall(est_mat .≈ log(1e-6))] .= -Inf
    for est in eachcol(est_mat)
        kvec = iprob.itf(est[1:n_rx])
        σs = exp.(est[n_rx+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)
        rxs = findall(isfinite.(est[1:n_rx]))
        push!(base_crns, sort(rxs))
    end
    base_crns_dict[settings] = base_crns
end

### Plots for papaer

## Directory for evaluation plots
eval_dir = joinpath(@__DIR__, "output/eval_figs");
mkpath(eval_dir);

## Top 20 CRNs
f = begin
    f = Figure(size=(1200, 1500), figure_padding=(5,5,5,20));
    cutoff_idx = 20
    xticks = 5:5:cutoff_idx
    for (j, k1) in enumerate(K1_VALS), (i, k18) in enumerate(reverse(K18_VALS))
        bic_merged = Dict{Vector{Int64},Float64}();
        logp_merged = Dict{Vector{Int64},Float64}();
        for pen_str in PEN_STRS
            bdict = bic_dict_lookup[(k1,k18,pen_str)]
            for (rxs, bic) in bdict
                if !haskey(bic_merged, rxs) || bic < bic_merged[rxs]
                    bic_merged[rxs] = bic
                    logp_merged[rxs] = -0.5*bic + logprior(length(rxs))
                end
            end
        end
        sort_bic_merged = sort(collect(bic_merged), by=(x)->x.second);
        sort_logp_merged = sort(collect(logp_merged), by=(x)->x.second, rev=true);
        n_merged = length(sort_logp_merged)

        logZ = logsumexp(last.(sort_logp_merged));
        post_merged =  Dict(rxs => exp(logj-logZ) for (rxs, logj) in sort_logp_merged);

        ax1 = Axis(
            f[2i-1,j],
            # title=i==1 ? L"k_1=%$k1" : "",
            ylabel=j==1 ? "Unnormalised log post." : "", 
            xlabel="CRNs sorted by post.", xticks=xticks, xlabelpadding=5.
        );
        ax2 = Axis(
            f[2i,j], alignmode=Mixed(top=5.), yreversed=true,
            yticks=(1:4, [j==1 ? L"$%$(pen_names_reg[pstr])$" : "" for pstr in PEN_STRS]), height=80, #yticklabelsize=16,
            limits=((nothing, nothing), (0.5, length(PEN_STRS)+0.5)), xticks=xticks, xaxisposition=:top, valign=:top,
            xminorgridcolor=(:black, .7), xminorgridvisible=true, xminorticksvisible=false, xminorticks=1.5:1:(cutoff_idx-0.5),
            yminorgridcolor=(:black, .7), yminorgridvisible=true, yminorticksvisible=false, yminorticks=1.5:1:(length(PEN_STRS)-0.5),
        )
        linkxaxes!(ax1, ax2)
        crns, logps = zip(sort_logp_merged[1:min(cutoff_idx, n_merged)]...)
        scatterlines!(ax1, collect(logps))
        for (i, pen_str) in enumerate(PEN_STRS)
            settings = (k1,k18,pen_str)
            post_dict = post_dict_lookup[settings]
            found_vec = haskey.(Ref(post_dict), crns)
            idxs = findall(found_vec)
            for idx in idxs
                highlight = crns[idx] ∈ [[true_rxs]; equiv_crns]
                b = band!(
                    ax2, [idx-0.5, idx+0.5], fill(i-0.5,2), fill(i+0.5,2), 
                    color=palette[i+2],
                    # color=highlight ? lighten(palette[i+2], 0.8) : darken(palette[i+2], 0.9)
                )
                translate!(b, 0, 0, -100)
            end
            comb_found_vec = .!(in.(first.(sort_logp_merged[idxs]), Ref(base_crns_dict[settings])))
            comb_idxs = idxs[findall(comb_found_vec)]
            scatter!(ax2, comb_idxs, fill(i, length(comb_idxs)), color=:grey20, marker=:star5)
            # size95 = min(length(top_95_dict[(k1,k18,pen_str)]), length(idxs))
            # scatter!(ax2, idxs[1:size95], fill(i, size95), color=:grey, marker=:star5)
            # Label(f[2i-1,6], L"k_{18}=%$(K18_VALS[i])", fontsize=16, halign=:left, tellheight=false)
        end 
        xlims!(0.5, cutoff_idx+0.5)
        rowgap!(f.layout, 2i-1, 5)
        colgap!(f.layout, 10)
        # colgap!(f.layout, 5, 5)
        Label(f[2i-1,j], L"k_1=%$(k1), \; k_{18}=%$(k18)", valign=:top, fontsize=16, tellheight=false, tellwidth=false)
    end
    f
end
save(joinpath(eval_dir, "top_crns.svg"), f);

## Plot all data
f = begin
    f = Figure(size=(1000, 1000), figure_padding=5);
    for (j, k1) in enumerate(K1_VALS), (i, k18) in enumerate(reverse(K18_VALS))
        ax = Axis(
            f[i,j],
            ylabel=j==1 ? "Concentration" : "", 
            xlabel=i==5 ? L"t" : "",
            title=i==1 ? L"k_1=%$k1" : "",
            #yscale=Makie.pseudolog10,
        );
        data_dir = get_data_dir(k1, k18);
        t_obs, data = read_data(joinpath(data_dir, "data.txt"));
        scale_fcts = scale_dict[(k1, k18)];
        iprob = make_iprob(
            oprob, k, t_obs, data, PEN_STRS[1], HYP_VALS[1]; 
            scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
        );
        true_kvec = make_true_kvec(k1, k18)
        oprob = remake(iprob.oprob, p=[k => true_kvec], u0=x0_map)
        sol_grid = solve(oprob)(t_grid).u
        for j in 1:3
            scatter!(ax, t_obs, data[j,:], color=(palette[j], 0.6), markersize=8)
            lines!(ax, t_grid, getindex.(sol_grid, j), color=darken(palette[j], 0.8), linewidth=2)
        end
    end
    for i in 1:5
        Label(f[i,6], L"k_{18}=%$(K18_VALS[i])", fontsize=16, halign=:left, tellheight=false)
    end
    colgap!(f.layout, 5, 5)
    Label(
        f[0,:], "Ground-truth trajectories and simulated data", font=:bold, fontsize=20,
    )
    Legend(
        f[6,:],
        [[
            LineElement(color=darken(palette[j],0.8), linestyle=nothing),
            MarkerElement(color=palette[j], marker=:circle)
        ] for j in 1:3],
        [L"X_{%$j}" for j in 1:3],
        labelsize=18,
        orientation=:horizontal
    )
    f
end
save(joinpath(eval_dir, "all_data.png"), f);


## Network reconstruction
tpos_dict = Dict(settings => length(rxs ∩ true_rxs) for (settings, rxs) in crn_mode_dict); # true positives
fpos_dict = Dict(settings => length(setdiff(rxs, true_rxs)) for (settings, rxs) in crn_mode_dict); # false positives
fneg_dict = Dict(settings => length(setdiff(true_rxs, rxs)) for (settings, rxs) in crn_mode_dict); # false negatives
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
                tpos_dict[settings]
            end for k1 in K1_VALS, k18 in K18_VALS]
        hm = heatmap!(mat, colormap=(:BuGn_3, 0.9), colorrange=(1, 3))
        for i in eachindex(K1_VALS), j in eachindex(K18_VALS)
            text!(ax, "$(mat[i, j])", position = (i, j), fontsize=16,
                color = :black, align = (:center, :center))
        end
    end
    Label(
        fig[0, 1:length(PEN_STRS)], "Number of ground-truth reactions in posterior mode CRN", 
        font = :bold, fontsize=20
    );
    Label(
        fig[2, 1:length(PEN_STRS)], "Number of spurious reactions in posterior mode CRN", 
        font = :bold, fontsize=20
    );
    # False positives
    fpos_max = maximum(
        begin
            settings = (k1, k18, pen_str);
            fpos_dict[settings]
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
                fpos_dict[settings]
            end for k1 in K1_VALS, k18 in K18_VALS]
        hm = heatmap!(mat, colormap=(:OrRd_3, 0.9), colorrange=(0, fpos_max), colorscale=sqrt)
        for i in eachindex(K1_VALS), j in eachindex(K18_VALS)
            text!(ax, "$(mat[i, j])", position = (i, j),
                color = :black, align = (:center, :center))
        end
    end
    # display(current_figure());
    save(joinpath(eval_dir, "eval_network.png"), fig);
end

## Posterior information of ground-truth CRN
min_prob = minimum(get(wdict, true_rxs, 1.0) for (_, wdict) in post_dict_lookup)
color_min = 10^floor(log10(min_prob))

begin
    fig = Figure(size=(1000, 300));
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
        
        hm_mat = [
            begin
                settings = (k1, k18, PEN_STRS[ax_j]);
                get(post_dict_lookup[settings], true_rxs, 1e-99)
            end for k1 in K1_VALS, k18 in K18_VALS]
        txt_mat = [
            begin
                settings = (k1, k18, PEN_STRS[ax_j]);
                sort_by_w = sort(collect(post_dict_lookup[settings]), rev=true, by=x->x.second);
                cutoff = findfirst((x)->(x.first==true_rxs), sort_by_w)
                cutoff === nothing ? "" : string(cutoff)
            end for k1 in K1_VALS, k18 in K18_VALS]
        hm = heatmap!(hm_mat, colormap=(:Blues_3, 0.9), colorrange=(color_min,1.), colorscale=log10, lowclip=(:grey,0.9))
        for i in eachindex(K1_VALS), j in eachindex(K18_VALS)
            text!(ax, "$(txt_mat[i, j])", position = (i, j), fontsize=16,
                color = :black, align = (:center, :center))
        end
        if ax_j == length(PEN_STRS)
            Colorbar(
                fig[1, end+1], hm, label="Posterior prob.", labelsize=16,
                labelrotation=π/2, ticklabelsize=16)
        end
    end
    Label(
        fig[0, 1:length(PEN_STRS)], "Posterior probability and rank of ground-truth CRN", 
        font = :bold, fontsize=20
    );
    # display(current_figure());
    save(joinpath(eval_dir, "true_prob.png"), fig);
end


## 95% HPD
max_avg_size = maximum(mean(length.(top_95)) for (_, top_95) in top_95_dict)
begin
    fig = Figure(size=(1000, 300));
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
        
        hm_mat = [
            begin
                settings = (k1, k18, PEN_STRS[ax_j]);
                mean(length.(top_95_dict[settings]))
            end for k1 in K1_VALS, k18 in K18_VALS]
        txt_mat = [
            begin
                settings = (k1, k18, PEN_STRS[ax_j]);
                length(top_95_dict[settings])
            end for k1 in K1_VALS, k18 in K18_VALS]
        hm = heatmap!(hm_mat, colormap=(:OrRd_3, 0.9), colorrange=(1, max_avg_size))
        for i in eachindex(K1_VALS), j in eachindex(K18_VALS)
            text!(ax, "$(txt_mat[i, j])", position = (i, j), fontsize=16,
                color = :black, align = (:center, :center))
        end
        if ax_j == length(PEN_STRS)
            Colorbar(
                fig[1, end+1], hm, label="Avg. number of reactions", labelsize=16,
                labelrotation=π/2, ticklabelsize=16, ticks=1:floor(max_avg_size))
        end
    end
    Label(
        fig[0, 1:length(PEN_STRS)], "Size and average number of reactions of 95% HPD sets", 
        font = :bold, fontsize=20
    );
    # display(current_figure());
    save(joinpath(eval_dir, "95_HPD.png"), fig);
end


## Rate constant estimation

function make_kerr_plot(k_idx, true_k_dict, fig, ax_i)
    k_err_dict = Dict(
        settings => begin
            isol = isol_dict_lookup[settings][crn]
            true_k = true_k_dict[settings];
            est_kvec = mask_kvec(isol, crn)
            (est_kvec[k_idx] - true_k) / true_k
        end for (settings, crn) in crn_mode_dict
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
            title = L"$%$(pen_names[PEN_STRS[ax_j]])$ \textbf{penalty}"
        )
        mat = [
            begin
                k_err_dict[(k1, k18, pen_str)]
            end for k1 in K1_VALS, k18 in K18_VALS]
        hm = heatmap!(mat, colormap=(:curl, 0.9), colorrange=(-1, 1))
    end
    Label(
        fig[ax_i-1, 1:length(PEN_STRS)], L"$\mathbf{k_{%$(k_idx)}} \textbf{ relative error for posterior mode CRN}$", 
        fontsize=20,
    )
end

true_k1_dict = Dict(settings[1:3] => settings[1] for settings in settings_vec);
true_k13_dict = Dict(settings[1:3] => 1.0 for settings in settings_vec);
true_k18_dict = Dict(settings[1:3] => settings[2] for settings in settings_vec);

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
    local settings = (k1, k18, pen_str)
    true_kvec = make_true_kvec(k1, k18);
    crn = crn_mode_dict[settings]
    isol = isol_dict_lookup[settings][crn]    
    est_kvec = mask_kvec(isol, crn_mode_dict[settings]) # zero out non-inferred reactions
    traj_err_dict[settings] = get_traj_err(
        est_kvec, true_kvec, isol.iprob.oprob, k, t_grid; u0=x0_map
    )
    pred_err_dict[settings] = get_traj_err(
        est_kvec, true_kvec, isol.iprob.oprob, k, t_grid; u0=x0_altmap
    )
end

traj_mean_dict = Dict(
    settings => traj_mean_err(traj_err)
    for (settings, traj_err) in traj_err_dict
);
traj_mean_min, traj_mean_max = extrema(values(traj_mean_dict))
pred_mean_dict = Dict(
    settings => traj_mean_err(pred_err)
    for (settings, pred_err) in pred_err_dict
);
pred_mean_min, pred_mean_max = extrema(values(pred_mean_dict))
colorrange = (min(traj_mean_min, pred_mean_min), max(traj_mean_max, pred_mean_max))

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
            title = L"$%$(pen_names[PEN_STRS[ax_i]])$ \textbf{penalty}"
        )
        
        mat = [
            begin
                traj_mean_dict[(k1, k18, pen_str)]
            end for k1 in K1_VALS, k18 in K18_VALS]
            hm = heatmap!(mat, colorscale=log10, colormap=(:Reds, 0.8), colorrange=colorrange)
    end;
    Label(
        fig[0, 1:length(PEN_STRS)], "Trajectory reconstruction error for posterior mode CRN", 
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
            title = L"$%$(pen_names[PEN_STRS[ax_i]])$ \textbf{penalty}"
        )
        
        mat = [
            begin
                pred_mean_dict[(k1, k18, pen_str)]
            end for k1 in K1_VALS, k18 in K18_VALS]
        hm = heatmap!(mat, colorscale=log10, colormap=(:Reds, 0.8), colorrange=colorrange)
        if ax_i == length(PEN_STRS)
            Colorbar(fig[:, end+1], hm, height = Relative(0.9), ticklabelsize=18)
        end
    end;
    Label(
        fig[2, 1:length(PEN_STRS)], "Trajectory prediction error for posterior mode CRN", 
        font = :bold, fontsize=20
    );
    save(joinpath(eval_dir, "traj_err.png"), fig);
end

## Print all reactions

ltx_vec = [
    begin 
        s = string(rx_vec[rx])
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => "&\\xrightarrow{k_{$rx}}")
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        # "\$$s\$"
        s
    end for rx in 1:n_rx
];
ltx_mat = reshape(ltx_vec, (10, 3));
ltx_lines = [join(row, " & ")*"\\\\" for row in eachrow(ltx_mat)];
println.(ltx_lines);


## Analyse 95% HPD

function contains_gt(crn)
    return ([1] ⊆ crn) && (any(alt ⊆ crn for alt in [[13],[11,14],[12,15]])) && (any(alt ⊆ crn for alt in [[18],[16,19],[17,20]]))
end

for k1 in K1_VALS, k18 in K18_VALS
    for pen_str in PEN_STRS
        println((k1, k18, pen_str))
        other_post = 0.0
        top_95 = top_95_dict[(k1, k18, pen_str)]
        for crn in top_95
            if contains_gt(crn) continue end
            # println(crn)
            other_post += post_dict_lookup[(k1, k18, pen_str)][crn]
        end
        other_post == 0.0 || println(other_post) # posterior prob. of other CRNs
    end
end

# values of dimensionless parameter estimates
dimless_ks = Vector{Float64}();
for k1 in K1_VALS, k18 in K18_VALS
    for pen_str in PEN_STRS
        top_95 = top_95_dict[(k1, k18, pen_str)]
        for crn in top_95
            est = isol_dict_lookup[(k1, k18, pen_str)][crn].est
            append!(dimless_ks, exp.(est[crn ∩ [1,13,18]]))
        end
    end
end
length(dimless_ks)
sort(dimless_ks)
hist(log10.(dimless_ks))
mean(dimless_ks .> 10)
mean(dimless_ks .< 1e-3)
mean(dimless_ks .< 1e-4)
extrema(dimless_ks)

## Temporary playground

k1 = 0.3; k18 = 3.;
data_dir = get_data_dir(k1, k18);
t_obs, data = read_data(joinpath(data_dir, "data.txt"));
scale_fcts = scale_dict[(k1, k18)];
iprob = make_iprob(
    oprob, k, t_obs, data, PEN_STRS[1], HYP_VALS[1]; 
    scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
);
bic_merged = Dict{Vector{Int64},Float64}();
logp_merged = Dict{Vector{Int64},Float64}();
for pen_str in PEN_STRS
    bdict = bic_dict_lookup[(k1,k18,pen_str)]
    for (rxs, bic) in bdict
        if !haskey(bic_merged, rxs) || bic < bic_merged[rxs]
            bic_merged[rxs] = bic
            logp_merged[rxs] = -0.5*bic + logprior(length(rxs))
        end
    end
end
sort_bic_merged = sort(collect(bic_merged), by=(x)->x.second);
sort_logp_merged = sort(collect(logp_merged), by=(x)->x.second, rev=true);
for i in 2:13
    println(sort_logp_merged[i])
end
tmp_crn = sort_logp_merged[3][1];
for pen_str in PEN_STRS
    println(haskey(isol_dict_lookup[(k1,k18,pen_str)], tmp_crn))
end
trg_isol = isol_dict_lookup[(k1,k18,"L1")][tmp_crn];
trg_logl = trg_isol.iprob.loss_func(trg_isol.kvec, trg_isol.σs)

pen_str = "logL1";
est_mat = reduce(hcat, [
    readdlm(joinpath(get_opt_dir(k1, k18, pen_str, hyp_val), "estimates.txt")) 
    for hyp_val in HYP_VALS]);
sorted_vec = [
    begin
        kvec = iprob.itf(est[1:n_rx])
        σs = exp.(est[n_rx+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)
        sort_reactions(isol, species_vec, rx_vec; abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false)
    end for est in eachcol(est_mat)
];
est_vec = [
    est for (est, sort_idxs) in zip(eachcol(est_mat), sorted_vec)
    if tmp_crn ⊆ sort_idxs[1:length(tmp_crn)]
];
for est in est_vec[end-10:end]
    kvec = iprob.itf(est[1:n_rx])
    σs = exp.(est[n_rx+1:end])
    isol = ODEInferenceSol(iprob, est, kvec, σs)
    sorted_idxs = sort_reactions(isol, species_vec, rx_vec; abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false)
    loss_val = isol.iprob.loss_func(isol.kvec, isol.σs)
    tmp_kvec = zeros(length(isol.kvec))
    idxs = sorted_idxs[1:length(tmp_crn)]
    @assert sort(idxs) == tmp_crn
    tmp_kvec[idxs] .= isol.kvec[idxs]
    fit_val = isol.iprob.loss_func(tmp_kvec, isol.σs)
    if fit_val - trg_logl < 10 && isol.kvec[19] > 1e-9
        global tmp_isol = isol
        println(fit_val - trg_logl)
        
    end
    # tmp_kvec = zeros(length(isol.kvec))
    # inferred = Vector{Int}()
    # for idx in sorted_idxs[1:5]
	# 	push!(inferred, idx)
    #     tmp_kvec[idx] = isol.kvec[idx]
	# 	fit_val = isol.iprob.loss_func(tmp_kvec, isol.σs)
	# 	loss_diff = fit_val - loss_val
	# 	println(idx, " ", loss_diff)	
    # end
    # println()
end

bic_dict = map_isol(tmp_isol, species_vec, rx_vec; abstol=1e-10, verbose=false, thres=10.)

for est in eachcol(est_mat)
    kvec = iprob.itf(est[1:n_rx])
    σs = exp.(est[n_rx+1:end])
    isol = ODEInferenceSol(iprob, est, kvec, σs)
    has_tmp_crn = any(issubset.(Ref(tmp_crn), keys(map_isol(isol, species_vec, rx_vec; abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false, thres=3))))
    if has_tmp_crn
        println(kvec[tmp_crn])
    end
end




# Posterior mean of number of reactions
for pen_str in PEN_STRS
    means = [
        sum(length(rxs)*w for (rxs, w) in post_dict_lookup[(k1,k18,pen_str)])
        for k1 in K1_VALS, k18 in K18_VALS
    ]
    display(extrema(means))
    display(sum(means .> 4.5))
end

[
    begin
        argmax(pen_str->sum(length(rxs)*w for (rxs, w) in post_dict_lookup[(k1,k18,pen_str)]), PEN_STRS)
    end for k1 in K1_VALS, k18 in K18_VALS
]

# HPD sizes
for pen_str in PEN_STRS
    sizes = [
        length(post_dict[(k1,k18,pen_str)])
        for k1 in K1_VALS, k18 in K18_VALS
    ]
    display(sort(collect(countmap(sizes))))
end

true_rxs = [1,13,18];
alts = [[1,11,14,18],[1,12,15,18],[1,13,16,19],[1,13,17,20]];
for trg_rxs in [[true_rxs]; alts]
println(trg_rxs)
in_post_vec = Vector{Int64}()
in_95_vec = Vector{Int64}()
mprob_vec = Vector{String}()
modds_vec = Vector{String}()
for pen_str in PEN_STRS
    # println(pen_str)
    in_95_ct = 0
    in_post_ct = 0
    wvec = Vector{Float64}() # posterior prob of alt
    ovec = Vector{Float64}() # posterior odds of alt to true
    for k1 in K1_VALS, k18 in K18_VALS
        settings = (k1, k18, pen_str)
        in_95 = trg_rxs ∈ top_95_dict[settings]
        in_95_ct += in_95
        in_post_ct += haskey(post_dict_lookup[settings], trg_rxs)

        n_95 = length(top_95_dict[settings])
        w = get(post_dict_lookup[settings], trg_rxs, 0.0)
        push!(wvec, w)
        wt = get(post_dict_lookup[settings], true_rxs, 0.0)
        o = w / wt
        push!(ovec, isnan(o) ? 0.0 : o)
        w_str = pyfmt(".4f", w)
        sort_by_w = sort(collect(post_dict_lookup[settings]), rev=true, by=x->x.second);
        w_rank = findfirst((x)->x.first==trg_rxs, sort_by_w)
        # println("$k1, $k18: $in_95, $w_rank out of $n_95, $w_str")
    end
    # println("$in_95_ct times in 95% HPD, $in_post_ct times found, median prob. $(median(wvec))")
    # display(sort(wvec))
    push!(in_post_vec,in_post_ct)
    push!(in_95_vec,in_95_ct)
    push!(mprob_vec, pyfmt(".4f", median(wvec)))
    push!(modds_vec, pyfmt(".4f", median(ovec)))
end
# println(join([in_post_vec; modds_vec], " & "))
println(join(in_post_vec, " & "))
println(join(in_95_vec, " & "))
end


for p1 in 1:4
    for p2 in (p1+1):4
        pen1, pen2 = PEN_STRS[p1], PEN_STRS[p2]
        tvds = Vector{Float64}()
        for k1 in K1_VALS, k18 in K18_VALS
            wdict1 = post_dict_lookup[(k1, k18, pen1)]
            wdict2 = post_dict_lookup[(k1, k18, pen2)]
            all_idxs = keys(wdict1) ∪ keys(wdict2)
            tvd = 0.5 * sum(abs(get(wdict1, idxs, 0.0)-get(wdict2, idxs, 0.0)) for idxs in all_idxs)
            push!(tvds, tvd)
            # println("$k1, $k18: $tvd")    
        end
        println("$pen1, $pen2: $(mean(tvds)), $(maximum(tvds))")  
    end
end

# does 95% HPD include ground-truth CRN
trg_rxs = [1,13,18];
for pen_str in PEN_STRS
    ct = 0
    for k1 in K1_VALS, k18 in K18_VALS
        settings = (k1,k18,pen_str)
        ct += trg_rxs ∈ top_95_dict[settings]
        if !(trg_rxs ∈ top_95_dict[settings])
            println(settings)
        end
    end
    println(ct)
end

# HPD coverage
trg_rxs = [1,13,18];
crits = Vector{Float64}();
pen_str = "logL1";
for k1 in K1_VALS, k18 in K18_VALS
    settings = (k1, k18, pen_str)
    sort_by_w = sort(collect(post_dict_lookup[settings]), rev=true, by=x->x.second);
    w_rank = findfirst((x)->x.first==trg_rxs, sort_by_w)
    push!(crits, sum(w for (_, w) in sort_by_w[1:w_rank]))
end

xs = range(0,1,101)
scatter(xs, [sum(c >= 1-x for c in crits)/25 for x in xs])
lines!([0,1],[0,1],color=:black,alpha=0.5,linestyle=:dash)
current_figure()


k1=1;k18=10;
tmp_sort = sort(
    collect(post_dict_lookup[(k1,k18,"logL1")]),
    by=last, rev=true
);
tmp_sort[1:10]

post_dict_lookup[(k1,k18,"logL1")][trg_rxs]
post_dict_lookup[(k1,k18,"approxL0")][trg_rxs]

bic_dict_lookup[(k1,k18,"logL1")][trg_rxs]
bic_dict_lookup[(k1,k18,"approxL0")][trg_rxs]

isol_dict_lookup[(k1,k18,"logL1")][trg_rxs].σs
isol_dict_lookup[(k1,k18,"approxL0")][trg_rxs].σs

101/2*sum(log, isol_dict_lookup[(k1,k18,"logL1")][trg_rxs].σs)
101/2*sum(log, isol_dict_lookup[(k1,k18,"approxL0")][trg_rxs].σs)

tmp_isol = isol_dict_lookup[(k1,k18,"logL1")][trg_rxs];
tmp_isol.iprob.loss_func(tmp_isol.kvec, tmp_isol.σs)

other_isol = isol_dict_lookup[(k1,k18,"approxL0")][trg_rxs];
tmp_isol.iprob.loss_func(other_isol.kvec, other_isol.σs)

σ_lbs = 0.01.*init_σs_dict[(k1, k18)]
std.(eachrow(tmp_isol.iprob.data))

# list CRNs in 95% HPD
k1=.1;k18=10.;
for pen_str in PEN_STRS
    println(pen_str)
    settings = (k1,k18,pen_str);
    for crn in top_95_dict[settings]
        println("$(crn => round(post_dict_lookup[settings][crn];digits=4))")
    end
end

for pen_str in PEN_STRS
    display(round(get(bic_dict_lookup[(k1,k18,pen_str)], [13,18,30], Inf);digits=4))
end

[
    rx_vec[i] => sum(w for (crn, w) in post_dict_lookup[(k1,k18,"hslike")] if i ∈ crn)
    for i in 1:n_rx
]

# posterior correlation / log odd ratio
k1=.1;k18=10.;
for pen_str in PEN_STRS
    settings = (k1,k18,pen_str);
    wpairs = collect(post_dict_lookup[settings])
    ws = ProbabilityWeights(last.(wpairs))
    hasrx_vec = [Int.(rx .∈ first.(wpairs)) for rx in 1:n_rx]
    println(pen_str)
    # println(log(y1y30)+log(n1n30)-log(y1n30)-log(n1y30))
    cmat = cor(reduce(hcat, hasrx_vec), ws)
    rxpairs = [(i,j) for j in 1:n_rx, i in 1:n_rx if i < j]
    sort_by_c = sort(rxpairs, by=(tpl)->cmat[tpl[1],tpl[2]])
    # display(current_figure())
    for (i,j) in sort_by_c
        if abs(cmat[i,j]) > 0.6
            println("$i $j $(cmat[i,j])")
        end
    end
end



k1=.1;k18=10.;
pen_str = "approxL0"
settings = (k1,k18,pen_str);
wdict = post_dict_lookup[settings];
marg_mat = [sum(w for (rxs, w) in wdict if rx ∈ rxs) for rx in 1:30]
near0 = findall(marg_mat .< 1e-6)

wpairs = collect(post_dict_lookup[settings]);
ws = ProbabilityWeights(last.(wpairs));
hasrx_vec = [Int.(rx .∈ first.(wpairs)) for rx in 1:n_rx]
println(pen_str)
# println(log(y1y30)+log(n1n30)-log(y1n30)-log(n1y30))
cmat = cor(reduce(hcat, hasrx_vec), ws)
rxpairs = [(i,j) for j in 1:n_rx, i in 1:n_rx if i < j];
sort_by_c = sort(rxpairs, by=(tpl)->cmat[tpl[1],tpl[2]]);

cmat[near0,:] .= NaN
cmat[:,near0] .= NaN
heatmap(cmat, colormap=:RdBu, colorrange=(-1,1))

fig = Figure()
ax = Axis(
    fig[1, 1], aspect=DataAspect(),
    xlabel = "Reaction index",
    ylabel = "Reaction index",
    title = L"Posterior reaction correlations (Approximate $L_0$)"
)

hm_mat = cmat;
hm = heatmap!(hm_mat, colormap=:RdBu, colorrange=(-1,1))
Colorbar(
    fig[1, end+1], hm, label="Correlation", labelsize=16,
    labelrotation=π/2, ticklabelsize=16)

display(current_figure())

minimum(d[[1,13,18]] for ((_,_,ps), d) in post_dict_lookup if ps != "L1")

# isol = isol_dict_lookup[settings][[1, 7, 13, 17]];
# isol.iprob.loss_func(isol.kvec, isol.σs)
# bic_dict_lookup[settings][[1, 7, 13, 17]]
# post_dict_lookup[settings][[1, 7, 13, 17]]


# isol = isol_dict_lookup[settings][[1, 13, 18]];
# isol.iprob.loss_func(isol.kvec, isol.σs)
# bic_dict_lookup[settings][[1, 13, 18]]
# post_dict_lookup[settings][[1, 13, 18]]



## Jaccard similarity of 95% HPD sets
# function calc_jac(x, y)
#     return length(x ∩ y) / length(x ∪ y)
# end

# for k1 in K1_VALS, k18 in K18_VALS
#     jac_mat = zeros(4,4)
#     for i in 1:4, j in 1:4
#         psi, psj = PEN_STRS[[i,j]]
#         jac_mat[i,j] = calc_jac(top_95_dict[(k1,k18,psi)], top_95_dict[(k1,k18,psj)])
#     end
#     println("$k1, $k18")
#     display(jac_mat)
# end

## Alternative to `evaluation.jl`

