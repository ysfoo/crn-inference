#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

using ProgressMeter
using LogExpFunctions, SpecialFunctions
using StatsBase

using StableRNGs

using Format
FMT_3DP = ".3f";

include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference
include(joinpath(@__DIR__, "../../src/plot_helper.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));
include(joinpath(@__DIR__, "setup.jl"));

# NO_CROSS = false
# EST_FNAME = NO_CROSS ? "nocross_estimates.txt" : "refined_estimates.txt";
# PLT_FNAME = NO_CROSS ? "traj_post_nocross.png" : "traj_post_crossover.png";

smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
scale_fcts = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k);

pen_names = Dict(
    "L1" => "\\mathbf{L_1}", 
    "logL1" => "\\textbf{log }\\mathbf{L_1}", 
    "approxL0" => "\\textbf{Approx. }\\mathbf{L_0}", 
    "hslike" => "\\textbf{Horseshoe}"
);

pen_names_reg = Dict(
    "L1" => "L_1",
    "logL1" => "\\text{log }L_1",
    "approxL0" => "\\text{Approx. }L_0",
    "hslike" => "\\text{Horseshoe}"
);

logprior(n) = -logabsbinomial(n_rx, n)[1] - log(n_rx+1)
# bb = elicit_bb_prior(n_rx, 6, 3)
# logprior(n) = -logabsbinomial(n_rx, n)[1] + logpdf(bb, n)


bic_dict_lookup = Dict{String, Dict{Vector{Int64},Float64}}();
post_dict_lookup = Dict{String, Dict{Vector{Int64},Float64}}();
isol_dict_lookup = Dict{String, Dict{Vector{Int64},ODEInferenceSol}}();
top_95_dict = Dict{String, Vector{Vector{Int64}}}();

iprob = make_iprob(
    oprob, k, t_obs, data, PEN_STRS[1], HYP_VALS[1]; 
    scale_fcts, abstol=1e-10, alg=KenCarp4(), verbose=false,
);
loss_func = iprob.loss_func;

for pen_str in PEN_STRS
    bic_by_rxs = Dict{Vector{Int64},Float64}();
    isol_by_rxs = Dict{Vector{Int64},ODEInferenceSol}();
    opt_dir = joinpath(@__DIR__, "output", pen_str)
    est_mat = readdlm(joinpath(opt_dir, "refined_estimates.txt"))
    est_mat[findall(est_mat .≈ log(1e-6))] .= -Inf
    for est in eachcol(est_mat)
        kvec = iprob.itf(est[1:n_rx])
        σs = exp.(est[n_rx+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)
        rxs = findall(isfinite.(est[1:n_rx]))
        bic = 2*loss_func(kvec, σs) + length(rxs)*log(length(data))
        if (get(bic_by_rxs, rxs, Inf) < bic) continue end
        bic_by_rxs[rxs] = bic
        isol_by_rxs[rxs] = isol
    end

    bic_dict_lookup[pen_str] = bic_by_rxs
    isol_dict_lookup[pen_str] = isol_by_rxs
    
    logps = Dict(rxs => -0.5*bic+logprior(length(rxs)) for (rxs, bic) in bic_dict_lookup[pen_str])
    logZ = logsumexp(collect(values(logps)))
    post_dict_lookup[pen_str] = posts = Dict(rxs => exp(logp-logZ) for (rxs, logp) in logps)

    cutoff_idx = findfirst((>)(0.95), cumsum(sort(collect(values(posts)), rev=true)))
    cutoff_logp = sort(collect(values(logps)), rev=true)[cutoff_idx]
    top_95_dict[pen_str] = [rxs for (rxs, logp) in logps if logp >= cutoff_logp]
end

# Merge results from all penalty functions
bic_merged = Dict{Vector{Int64},Float64}();
logp_merged = Dict{Vector{Int64},Float64}();
for (pen_str, bdict) in bic_dict_lookup
    for (rxs, bic) in bdict
        if !haskey(bic_merged, rxs) || bic < bic_merged[rxs]
            bic_merged[rxs] = bic
            logp_merged[rxs] = -0.5*bic + logprior(length(rxs))
        end
    end
end
sort_bic_merged = sort(collect(bic_merged), by=(x)->x.second);
sort_logp_merged = sort(collect(logp_merged), by=(x)->x.second, rev=true);

logZ = logsumexp(last.(sort_logp_merged));
post_merged =  Dict(rxs => exp(logj-logZ) for (rxs, logj) in sort_logp_merged);

base_crns_dict = Dict{String, Set{Vector{Int64}}}()
for pen_str in PEN_STRS
    base_crns = Set{Vector{Int64}}();
    opt_dir = joinpath(@__DIR__, "output", pen_str)
    est_mat = readdlm(joinpath(opt_dir, "nocross_estimates.txt"))
    est_mat[findall(est_mat .≈ log(1e-6))] .= -Inf
    for est in eachcol(est_mat)
        kvec = iprob.itf(est[1:n_rx])
        σs = exp.(est[n_rx+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)
        rxs = findall(isfinite.(est[1:n_rx]))
        push!(base_crns, sort(rxs))
    end
    base_crns_dict[pen_str] = base_crns
end

# TVD
function calc_tvd(wdict1, wdict2)
    all_idxs = keys(wdict1) ∪ keys(wdict2)
    return 0.5 * sum(abs(get(wdict1, idxs, 0.0)-get(wdict2, idxs, 0.0)) for idxs in all_idxs)
end
for pen_str in PEN_STRS
    tvd = calc_tvd(post_dict_lookup[pen_str], post_merged)
    println("$pen_str: $tvd")
end

### Results for paper

fig_dir = joinpath(@__DIR__, "output/eval_figs");
mkpath(fig_dir);

# Plot trajectories and log posterior probability (unnormalised)
t_grid = range(t_span..., 500);
PLOT_NEW = true
begin
    pen_str = "logL1"
    right_range = 4:5
    msize = 8
    datac = (:grey40, 0.9)
    c = palette[2+findfirst((==)(pen_str), PEN_STRS)]
    f = Figure(size=(1000, 500), figure_padding=5);
    xticks = 5:5:25
    for i in 1:n_species
        ax_i = div(i-1, 3) + 1
        ax_j = mod(i-1, 3) + 1
        ax = Axis(f[ax_i, ax_j], xlabel=i<=2 ? "" : L"$t$ ($10^4$ min)", ylabel=L"$x_{%$i}(t)$", xtickformat="{:.0f}")
        scatter!(t_obs ./ 1e4, data[i,:], color=datac, markersize=msize)
        # lines!(t_grid, getindex.(gold_osol.u, i), color=(palette[2],0.9), label="Gold std.")
        sort_by_w = sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
        for rxs in top_95_dict[pen_str]
        # for (rxs, _) in sort_by_w[1:25]
            isol = isol_dict_lookup[pen_str][rxs];
            remade_oprob = remake(iprob.oprob, p=[k=>isol.kvec]);
            osol = solve(remade_oprob, saveat=t_grid);
            lines!(
                t_grid ./ 1e4, getindex.(osol.u, i), color=(c,0.5), linewidth=1
            )
        end
    end
    ax = Axis(f[1:2,1:3], title="Pinene isomerisation data and\nestimated trajectories", titlesize=18)
    hidedecorations!(ax)
    hidespines!(ax)
    cutoff_idx = 25
    if PLOT_NEW
        ax1 = Axis(
            f[1:2,right_range], title="Top $cutoff_idx CRNs across\nall penalty functions", titlesize=18,
            ylabel="Unnormalised log posterior", 
            xlabel="CRNs sorted by posterior probability", xticks=xticks, xlabelpadding=5.
        );
        ax2 = Axis(
            f[3,right_range], alignmode=Mixed(top=5.), yreversed=true,
            yticks=(1:4, [L"$%$(pen_names_reg[pstr])$" for pstr in PEN_STRS]), yticklabelsize=16, height=80,
            limits=((nothing, nothing), (0.5, length(PEN_STRS)+0.5)), xticks=xticks, xaxisposition=:top, valign=:top,
            xminorgridcolor=(:black, .7), xminorgridvisible=true, xminorticksvisible=false, xminorticks=1.5:1:(cutoff_idx-0.5),
            yminorgridcolor=(:black, .7), yminorgridvisible=true, yminorticksvisible=false, yminorticks=1.5:1:(length(PEN_STRS)-0.5),
        )
        linkxaxes!(ax1, ax2)
        logps = last.(sort_logp_merged[1:cutoff_idx]);
        scatterlines!(ax1, logps)
        for (i, pen_str) in enumerate(PEN_STRS)
            post_dict = post_dict_lookup[pen_str]
            found_vec = haskey.(Ref(post_dict), first.(sort_logp_merged[1:cutoff_idx]))
            idxs = findall(found_vec)
            for idx in idxs
                crn = sort_logp_merged[idx].first
                # scatter!(
                #     ax1, idx-0.25+0.1*i, -0.5*bic_dict_lookup[pen_str][crn]+logprior(length(crn)),
                #     color=(palette[i+2],0.8), marker=[:rect,:diamond,:cross,:xcross][i]
                # )
                b = band!(ax2, [idx-0.5, idx+0.5], fill(i-0.5,2), fill(i+0.5,2), color=palette[i+2])
                translate!(b, 0, 0, -100)
            end
            comb_found_vec = .!(in.(first.(sort_logp_merged[idxs]), Ref(base_crns_dict[pen_str])))
            comb_idxs = idxs[findall(comb_found_vec)]
            scatter!(ax2, comb_idxs, fill(i, length(comb_idxs)), color=:grey20, marker=:star5)
            # size95 = min(length(top_95_dict[pen_str]), length(idxs))
            # scatter!(ax2, idxs[1:size95], fill(i, size95), color=:grey, marker=:star5)
        end       
        xlims!(0.5, cutoff_idx+0.5)
    else
        ax = Axis(
            f[1:2,4:5], title="Unnormalised log posterior\nfor top $cutoff_idx CRNs", titlesize=18,
            ylabel="Unnormalised log posterior", 
            xlabel="CRNs sorted by posterior", xticks=xticks
        );
        for (i, pen_str) in enumerate(PEN_STRS)
            posts = [-0.5*bic+logprior(length(inferred)) for (inferred, bic) in bic_dict_lookup[pen_str]];
            scatterlines!(
                sort(posts, rev=true)[1:cutoff_idx], color=(palette[i+2],0.8), 
                marker=[:rect,:diamond,:cross,:xcross][i],
                label=L"$%$(pen_names_reg[pen_str])$"
            )
        end
        xlims!(0, cutoff_idx)
        f[3,4:5] = Legend(f, ax, nbanks=2, tellheight=true, labelsize=16);
    end
    f[3,1:3] = Legend(
        f, [MarkerElement(color=datac, marker=:circle, markersize=msize), LineElement(color=c)], 
        ["Data", L"CRNs in 95% HPD set ($%$(pen_names_reg[pen_str])$ penalty)"],
        orientation=:horizontal, labelsize=16, tellheight=false
    )
    rowgap!(f.layout, 1, 0)
    rowgap!(f.layout, 2, 5)
    Label(
        f[1, 1, TopLeft()], "A",
        fontsize = 20,
        font = :bold,
        padding = (20, 0, 35, 0),
        halign = :right
    )
    Label(
        f[1, 4, TopLeft()], "B",
        fontsize = 20,
        font = :bold,
        padding = (20, 0, 35, 0),
        halign = :right
    )
    f
end
save(joinpath(fig_dir, "top_crns.png"), f);

f = begin
    f = Figure(size=(800, 800), figure_padding=5);
    for (i, pen_str) in enumerate(PEN_STRS)
        wpairs = collect(post_dict_lookup[pen_str]);
        ws = ProbabilityWeights(last.(wpairs));
        hasrx_vec = [Int.(rx .∈ first.(wpairs)) for rx in 1:n_rx];
        cmat = cor(reduce(hcat, hasrx_vec), ws);

        ax_i = div(i-1, 2) + 1
        ax_j = mod(i-1, 2) + 1
        ax = Axis(
            f[ax_i,ax_j],
            title=L"%$(pen_names_reg[pen_str])\text{ penalty}",
            titlesize=18,
            aspect=DataAspect(), 
            xlabel="Reaction index", ylabel="Reaction index",
            xlabelsize=18, ylabelsize=18,
            xticks=(1:n_rx, string.(1:n_rx)), 
            yticks=(1:n_rx, string.(1:n_rx)), 
        )
        heatmap!(ax, cmat, colormap=:RdBu, colorrange=(-1,1))        
    end
    Label(f[1, 1:2, Top()], "Posterior correlations", valign = :bottom,
    font = :bold, fontsize=20,
    padding = (0, 0, 20, 0))
    Colorbar(
    f[1:2,3], ticklabelsize=16, colormap=:RdBu, colorrange=(-1,1), height=Relative(0.7),
    )
    colgap!(f.layout, 5)
    f
end
save(joinpath(fig_dir, "corrs.png"), f);

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
for i in [7,8,15,16]
    insert!(ltx_vec, i, " & ")
end
ltx_mat = reshape(ltx_vec, (8, 3));
ltx_lines = [join(row, " & ")*"\\\\" for row in eachrow(ltx_mat)];
println.(ltx_lines);

rx_vec
rx_subset = [
    1, 2, 9, 19, 15, 12, 17, 5, 14
];

# convert reactions to latex
println.([
    begin 
        s = string(rx_vec[rx])
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => "\\rightarrow")
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        "\$$s\$ & & & &"
    end for rx in rx_subset
]);

marg_mat = [
    begin
        wdict = post_dict_lookup[pen_str]
        sum(w for (rxs, w) in wdict if rx ∈ rxs)
    end for rx in rx_subset, pen_str in PEN_STRS
];

for row in eachrow(marg_mat)
    println(join(pyfmt.(Ref(".4f"), row), " & "))
end

other_subset = sort(setdiff(1:n_rx, rx_subset));
others_mat = [
    begin
        wdict = post_dict_lookup[pen_str]
        sum(w for (rxs, w) in wdict if rx ∈ rxs)
    end for rx in other_subset, pen_str in PEN_STRS
]

# size of 95% HPD
[p => length(top_95) for (p, top_95) in top_95_dict]
[p => reduce(intersect, top_95) for (p, top_95) in top_95_dict]

has9 = findfirst(x->9∈x.first, sort_logp_merged)
post_merged[sort_logp_merged[has9].first]

exit()

### Visualise reactions of top 25
begin
f = Figure();
ax = Axis(f[1,1]);
for i in 1:25
    crn = sort_logp_merged[i].first
    for pen_str in PEN_STRS
        if haskey(isol_dict_lookup[pen_str], crn)
            isol = isol_dict_lookup[pen_str][crn]
            for r in crn
                band!(
                    ax, [i-0.5,i+0.5], [r-0.5], [r+0.5], 
                    color=fill(isol.est[r],2),
                    colormap=:Blues, colorrange=log.([1e-6,1e2])
                )
            end
            break
        end
    end
    scatter!(ax, fill(i,length(crn)), crn, color=:black)
end
f
end

for i in 1:25
    for j in 1:(i-1)
        crn1, crn2 = sort_logp_merged[i].first, sort_logp_merged[j].first
        if (crn1 ⊆ crn2) println((i, j)) end
        if (crn2 ⊆ crn1) println((j, i)) end
    end
end

lines(length.(first.(sort_logp_merged)))

lines(length.(first.(sort_logp_merged[1:500])))

setdiff(sort_logp_merged[12].first, sort_logp_merged[4].first)
exp.(isol_dict_lookup[pen_str][sort_logp_merged[12].first].est[[11,14]])


### Look at CRNs that are not shared

iprob = make_iprob(
    oprob, k, t_obs, data, PEN_STRS[1], HYP_VALS[1]; 
    scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
);
sort_bic_merged = sort(collect(bic_merged), by=(x)->x.second);
sort_logp_merged = sort(collect(logp_merged), by=(x)->x.second, rev=true);

trg = sort_logp_merged[5].first
println(trg)
logp_merged[trg]
exp(logp_merged[trg]-logZ)
exp.(isol_dict_lookup["L1"][trg].est[trg])
exp.(isol_dict_lookup["logL1"][trg].est[trg])
exp.(isol_dict_lookup["approxL0"][trg].est[trg])
exp.(isol_dict_lookup["hslike"][trg].est[trg])

exp.(isol_dict_lookup["L1"][trg].σs)
exp.(isol_dict_lookup["logL1"][trg].σs)
exp.(isol_dict_lookup["approxL0"][trg].σs)
exp.(isol_dict_lookup["hslike"][trg].σs)

bic_dict_lookup["L1"][trg]
bic_dict_lookup["logL1"][trg]
bic_dict_lookup["approxL0"][trg]
bic_dict_lookup["hslike"][trg]

for i in 1:25
    println(sort_logp_merged[i])
    crn = sort_logp_merged[i].first
    for pen_str in PEN_STRS
        if !haskey(isol_dict_lookup[pen_str], crn)
            continue
        end
        println("$pen_str $(minimum(exp.(isol_dict_lookup[pen_str][crn].est[crn])) > 1e-4)")
    end
end

function find_diff_pair(crn1, crn2)
    return Pair(setdiff(crn1, crn2), setdiff(crn2, crn1))
end

function jac_sim(crn1, crn2)
    return length(crn1 ∩ crn2) / length(crn1 ∪ crn2)
end

function translate_crn(crn, crn_diff)
    sort(setdiff(crn, crn_diff.second) ∪ crn_diff.first)
end

pen_str = "hslike"
# map (alt reactions, replaced reactions) to CRNs with alt reactions
max_diffs = 100;
crn_diffs = Dict{Pair{Vector{Int64},Vector{Int64}}, Vector{Vector{Int64}}}();
max_post = maximum(values(post_dict_lookup[pen_str]));
sort_by_post = sort(collect(post_dict_lookup[pen_str]), by=(x)->x.second, rev=true);
n_crns = length(sort_by_post)
crn_pairs = sort(
    vec(collect(Iterators.product(1:n_crns, 1:n_crns))), 
    rev=true, by=(x)->sort_by_post[x[1]].second*sort_by_post[x[2]].second
);
for (i, j) in crn_pairs
    crn1, crn2 = sort_by_post[i].first, sort_by_post[j].first
    crn_diff = find_diff_pair(crn1, crn2)
    if (length(crn_diff.first) <= 2 && length(crn_diff.second) <= 2)
        haskey(crn_diffs, crn_diff) || begin crn_diffs[crn_diff] = [] end
        push!(crn_diffs[crn_diff], crn1)
        length(crn_diffs) >= max_diffs && break
    end
end
length(crn_diffs)

n_top = clamp(findall(x->x.second/max_post>1e-4, sort_by_post)[end], 25, 100)
top_crns = first.(sort_by_post)[1:n_top];
crn_cands = Set{Vector{Int64}}();
tmp = 0;
for crn in top_crns
    for (crn_diff, crns) in crn_diffs
        crn_diff.second ⊆ crn || continue
        tmp += length(crn_diff.first) > 0 ? length(crns) : 1
        push!(crn_cands, translate_crn(crn, crn_diff))
    end
end
length(crn_cands), tmp
# findfirst((x)->!(x∈crn_cands), first.(sort_logp_merged))
findall((x)->!(x∈crn_cands), first.(sort_logp_merged))[1:10]
sort_logp_merged[1:10]
sort_by_post[1:10]

for p1 in 1:4
    for p2 in (p1+1):4
        pen1, pen2 = PEN_STRS[p1], PEN_STRS[p2]
        wdict1 = post_dict_lookup[pen1]
        wdict2 = post_dict_lookup[pen2]
        tvd = calc_tvd(wdict1, wdict2)
        println("$pen1, $pen2: $tvd")  
    end
end

# Combine all solutions

bic_merged = Dict{Vector{Int64},Float64}();
for (pen_str, bdict) in bic_dict_lookup
    for (rxs, bic) in bdict
        if !haskey(bic_merged, rxs) || bic < bic_merged[rxs]
            bic_merged[rxs] = bic
        end
    end
end

# Check if BICs for same CRN across different penalty funcs are equal
sort_bic_merged = sort(collect(bic_merged), by=(x)->x.second);
for (crn , _)in sort_bic_merged[1:10]
    println(crn)
    for pen_str in PEN_STRS
        bdict = bic_dict_lookup[pen_str]
        display((round(get(bdict, crn, Inf); digits=4), pen_str))
    end
end

crn = trg;
isol_dict_lookup["approxL0"][crn].σs
isol_dict_lookup["hslike"][crn].σs
tmp_func = make_refine_func(iprob, crn, fill(0.05, n_species), fill(5., n_species));

tmp_u1 = collect(isol_dict_lookup["logL1"][crn].est)[[crn;21:25]];
tmp_u2 = collect(isol_dict_lookup["approxL0"][crn].est)[[crn;21:25]];
tmp_u3 = collect(isol_dict_lookup["hslike"][crn].est)[[crn;21:25]];

using ForwardDiff, LinearAlgebra, BenchmarkTools
cfg = ForwardDiff.GradientConfig(tmp_func, tmp_u1);
norm(ForwardDiff.gradient(tmp_func, tmp_u1, cfg))
norm(ForwardDiff.gradient(tmp_func, tmp_u2, cfg))
norm(ForwardDiff.gradient(tmp_func, tmp_u3, cfg))
ForwardDiff.gradient(tmp_func, tmp_u1, cfg)
@btime tmp_func($tmp_u1)
@btime tmp_func($tmp_u2)
@btime tmp_func($tmp_u3)

refined_isol, refined_res = refine_isol(
    tmp_func, iprob, isol_dict_lookup["hslike"][crn].est, crn,
    σ_lbs, σ_lbs;
    optim_alg=BFGS(),
    # optim_opts=Optim.Options(iterations=100, x_abstol=1e-10, f_abstol=1e-10, outer_x_abstol=1e-10, outer_f_abstol=1e-10),
    optim_opts=Optim.Options(
        iterations=10^5
        # iterations=100, outer_iterations=100, f_calls_limit=100000, time_limit=600.,
        # x_abstol=1e-10, f_abstol=1e-10, outer_x_abstol=1e-10, outer_f_abstol=1e-10,
    ),    
);

refined_res

# Compare BIC across penalty functions

for (rxs, post) in sort(collect(post_merged), rev=true, by=(x)->x.second)
    if post < 1e-3
        break
    end
    display((rxs, round(post;digits=4)))
    for pen_str in PEN_STRS
        bic_dict = bic_dict_lookup[pen_str]
        display(round(get(bic_dict, rxs, -Inf);digits=4))
    end
end


pen_str = "hslike"
top_95_dict[pen_str]
trg_rxs = [1, 2, 19, 15, 12, 17, 5, 14]
for rxs in top_95_dict[pen_str]
    ilen = length(trg_rxs ∩ rxs)
    ulen = length(trg_rxs ∪ rxs)
    println(ilen, " ", length(rxs)-ilen)
end


for pen_str in PEN_STRS
    println(reduce(∩, top_95_dict[pen_str]))
end

for pen_str in PEN_STRS
    posts = Dict(rxs => -0.5*bic+logprior(length(rxs)) for (rxs, bic) in bic_dict_lookup[pen_str]);
    logZ = logsumexp(collect(values(posts)));
    display(logZ)
end

sortw_dict = Dict(
    pen_str => sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second) for pen_str in PEN_STRS
);
sortw_dict["logL1"][1:5]
sortw_dict["approxL0"][1:5]
sortw_dict["hslike"][1:5]

findfirst((x)->x.first==sortw_dict["logL1"][1].first, sortw_dict["approxL0"])

sum(log, isol_dict_lookup["approxL0"][sortw_dict["approxL0"][1].first].σs)

over_thres = [];
for (rxs, bic) in bic_merged
    if -0.5*bic+logprior(length(rxs)) > 13
        push!(over_thres, rxs)
    end
end
println(reduce(∩, over_thres))

rx_vec

# posterior correlation / log odd ratio
for pen_str in PEN_STRS
    wpairs = collect(post_dict_lookup[pen_str])
    ws = ProbabilityWeights(last.(wpairs))
    hasrx_vec = [Int.(rx .∈ first.(wpairs)) for rx in 1:n_rx]
    println(pen_str)
    # println(log(y1y30)+log(n1n30)-log(y1n30)-log(n1y30))
    cmat = cor(reduce(hcat, hasrx_vec), ws)
    rxpairs = [(i,j) for j in 1:n_rx, i in 1:n_rx if i < j]
    sort_by_c = sort(rxpairs, by=(tpl)->cmat[tpl[1],tpl[2]])
    n = length(sort_by_c)
    for (i,j) in sort_by_c
        if abs(cmat[i,j]) > 0.5
            println("$i $j $(cmat[i,j])")
        end
    end
end

# 5 | 6 12
# 5     : 2->3
# 6     : 2->4
# 12    : 4->3

# 5 | 11 12
# 5     : 2->3
# 11    : 4->2
# 12    : 4->3

# 1 (14|15) | 6 11
# 1     : 1->2
# 6     : 2->4
# 11    : 4->2

# Posterior mean of number of reactions
for pen_str in PEN_STRS
    mean_size = sum(length(rxs)*w for (rxs, w) in post_dict_lookup[pen_str])
    display(mean_size)
end

length(reduce(intersect, values(top_95_dict)))
length(reduce(intersect, [top_95_dict[p] for p in PEN_STRS[2:4]]))
length(reduce(intersect, [top_95_dict[p] for p in PEN_STRS[[1,3,4]]]))
length(reduce(intersect, [top_95_dict[p] for p in PEN_STRS[[1,2,4]]]))
length(reduce(intersect, [top_95_dict[p] for p in PEN_STRS[1:3]]))

length(top_95_dict["logL1"] ∩ top_95_dict["approxL0"])

[p => length(top_95) for (p, top_95) in top_95_dict]

# Sum all computation times
# tot_time = 0.0;
# for pen_str in PEN_STRS
#     for hyp_val in HYP_VALS
#         opt_dir = get_opt_dir(pen_str, hyp_val);
#         file_time = open(joinpath(opt_dir, "optim_progress.txt")) do io
#             sum(parse(Float64, collect(eachsplit(line))[3]) for line in eachline(io))
#         end
#         tot_time += file_time
#     end
# end
# tot_time