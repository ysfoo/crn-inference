#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

using ProgressMeter
using LogExpFunctions
using SpecialFunctions
using StatsBase

using StableRNGs

using Format
FMT_3DP = ".3f";

READ_GOLD = true;

include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference
include(joinpath(@__DIR__, "../../src/plot_helper.jl"));
include(joinpath(@__DIR__, "../eval_helper.jl"));
include(joinpath(@__DIR__, "setup.jl"));
include(joinpath(@__DIR__, "gold_std.jl"));

NO_CROSS = false
EST_FNAME = NO_CROSS ? "nocross_estimates.txt" : "refined_estimates.txt"
PLT_FNAME = NO_CROSS ? "top_crns_nocross.png" : "top_crns_crossover.png"

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

bb = elicit_bb_prior(n_rx, 20, 5)
alt_logprior(n) = -logabsbinomial(n_rx, n)[1] + logpdf(bb, n)

calc_jac(x, y) = length(x ∩ y) / length(x ∪ y);

bic_dict_lookup = Dict{String, Dict{Vector{Int64},Float64}}();
post_dict_lookup = Dict{String, Dict{Vector{Int64},Float64}}();
isol_dict_lookup = Dict{String, Dict{Vector{Int64},ODEInferenceSol}}();
top_95_dict = Dict{String, Vector{Vector{Int64}}}();
refine_func_dict = Dict{Vector{Int64}, FunctionWrapper}();

iprob = make_iprob(
    oprob, k, t_obs, data, PEN_STRS[1], HYP_VALS[1]; 
    scale_fcts, alg=AutoVern7(KenCarp4()), abstol=1e-10, verbose=false
);
loss_func = iprob.loss_func;

for pen_str in PEN_STRS
    bic_by_rxs = Dict{Vector{Int64},Float64}();
    isol_by_rxs = Dict{Vector{Int64},ODEInferenceSol}();
    opt_dir = joinpath(@__DIR__, "output", pen_str)
    est_mat = readdlm(joinpath(opt_dir, EST_FNAME))
    @assert allunique(findall(isfinite.(est[1:n_rx])) for est in eachcol(est_mat))
    for est in eachcol(est_mat)
        kvec = iprob.itf(est[1:n_rx])
        σs = exp.(est[n_rx+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)
        rxs = findall(isfinite.(est[1:n_rx]))
        bic = 2*loss_func(kvec, σs) + length(rxs)*log(length(data))
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
sum(post_merged[crn] for crn in first.(sort_logp_merged[1:5]))

# TVD
function calc_tvd(wdict1, wdict2)
    all_idxs = keys(wdict1) ∪ keys(wdict2)
    return 0.5 * sum(abs(get(wdict1, idxs, 0.0)-get(wdict2, idxs, 0.0)) for idxs in all_idxs)
end
for pen_str in PEN_STRS
    tvd = calc_tvd(post_dict_lookup[pen_str], post_merged)
    println("$pen_str: $tvd")
end

# Gold standard
gold_bic = -2*gold_logp + length(gold_idxs)*log(length(data))
gold_post = -0.5*gold_bic + logprior(length(gold_idxs))

### Results for paper

fig_dir = joinpath(@__DIR__, "output/eval_figs");
mkpath(fig_dir);

# Plot trajectories
PLOT_NEW = true
begin
    pen_str = "logL1"
    goldc = :black
    c = palette[2+findfirst((==)(pen_str), PEN_STRS)]
    f = Figure(size=(1000, 500));
    xticks = 5:5:25
    for i in 1:n_species
        ax_i = div(i-1, 3) + 1
        ax_j = mod(i-1, 3) + 1
        ax = Axis(f[ax_i, ax_j], xlabel=L"$t$", ylabel=L"$x_{%$i}(t)$", xtickformat="{:.0f}")
        scatter!(t_obs, data[i,:], color=(:grey, 0.9))
        sort_by_w = sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
        # for rxs in top_95_dict[pen_str]
        for (rxs, _) in sort_by_w[1:25]
            isol = isol_dict_lookup[pen_str][rxs];
            remade_oprob = remake(isol.iprob.oprob, p=[k=>isol.kvec]);
            osol = solve(remade_oprob, saveat=t_grid);
            lines!(
                t_grid, getindex.(osol.u, i), color=(c,0.5), linewidth=1
            )
        end
        lines!(t_grid, getindex.(gold_osol.u, i), color=(goldc,0.8), linestyle=(:dash,:dense), linewidth=1.5)
    end
    ax = Axis(f[1:2,1:3], title="(a) Pyridine denitrogenation data and\nestimated trajectories", titlesize=18)
    hidedecorations!(ax)
    hidespines!(ax)
    cutoff_idx = 25
    if PLOT_NEW
        ax1 = Axis(
            f[1:2,4:5], title="(b) Top $cutoff_idx CRNs across\nall penalty functions", titlesize=18,
            ylabel="Unnormalised log posterior", 
            xlabel="CRNs sorted by posterior probability", xticks=xticks, xlabelpadding=5.
        );
        ax2 = Axis(
            f[3,4:5], alignmode=Mixed(top=5.), yreversed=true,
            yticks=(1:4, [L"$%$(pen_names_reg[pstr])$" for pstr in PEN_STRS]), yticklabelsize=16, height=80,
            limits=((nothing, nothing), (0.5, length(PEN_STRS)+0.5)), xticks=xticks, xaxisposition=:top, valign=:top,
            xminorgridcolor=(:black, .7), xminorgridvisible=true, xminorticksvisible=false, xminorticks=1.5:1:(cutoff_idx-0.5),
            yminorgridcolor=(:black, .7), yminorgridvisible=true, yminorticksvisible=false, yminorticks=1.5:1:(length(PEN_STRS)-0.5),
        )
        linkxaxes!(ax1, ax2)
        logps = last.(sort_logp_merged[1:cutoff_idx]);
        scatterlines!(ax1, logps)
        lines!(ax1, [0, cutoff_idx], fill(gold_post, 2), color=(goldc, 0.8), linestyle=:dash, label="Gold standard")
        for (i, pen_str) in enumerate(PEN_STRS)
            # scatter!(ax2, (1:5).+i, fill(i, 5), color=palette[1])
            post_dict = post_dict_lookup[pen_str]
            found_vec = haskey.(Ref(post_dict), first.(sort_logp_merged[1:cutoff_idx]))
            idxs = findall(found_vec)
            for idx in idxs
                b = band!(ax2, [idx-0.5, idx+0.5], fill(i-0.5,2), fill(i+0.5,2), color=(palette[i+2],0.8))
                translate!(b, 0, 0, -100)
            end
            size95 = min(length(top_95_dict[pen_str]), length(idxs))
            scatter!(ax2, idxs[1:size95], fill(i, size95), color=:grey, marker=:star5)
        end
        xlims!(0.5, cutoff_idx+0.5)
    else
        ax = Axis(
            f[1:2,4:5], title="(b) Unnormalised log posterior\nfor top $cutoff_idx CRNs", titlesize=18,
            ylabel="Unnormalised log posterior", 
            xlabel="CRNs sorted by posterior"
        );
        for (i, pen_str) in enumerate(PEN_STRS)
            posts = [-0.5*bic+logprior(length(inferred)) for (inferred, bic) in bic_dict_lookup[pen_str]];
            scatterlines!(
                sort(posts, rev=true)[1:cutoff_idx], color=(palette[i+2],0.8), 
                marker=[:rect,:diamond,:cross,:xcross][i],
                label=L"$%$(pen_names_reg[pen_str])$"
            )
        end
        lines!([0, cutoff_idx], fill(gold_post, 2), color=(goldc, 0.8), linestyle=:dash, label="Gold standard")
        xlims!(0, cutoff_idx)
        f[3,4:5] = Legend(f, ax, nbanks=2, tellheight=true, labelsize=16);
    end
    f[3,1:3] = Legend(
        f, [MarkerElement(color=(:grey, 0.9), marker=:circle), LineElement(color=c), LineElement(color=goldc, linestyle=(:dash,:dense))], 
        ["Data", L"Top 25 CRNs obtained with $%$(pen_names_reg[pen_str])$ penalty", "Gold standard"],
        orientation=:horizontal, labelsize=16, tellheight=false,
    )
    rowgap!(f.layout, 2, 5)
    f
end
save(joinpath(fig_dir, PLT_FNAME), f);

## Print all reactions with boxes 
ltx_vec = [
    begin 
        s = string(rx_vec[rx])
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => "&\\xrightarrow{k_{$rx}}")
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        # "\$$s\$"
        rx ∈ gold_idxs ? "\\tikzmark{start$rx} $s \\tikzmark{end$rx}" : s
    end for rx in 1:n_rx
];
for i in [68]
    insert!(ltx_vec, i, " & ")
end
ltx_mat = reshape(ltx_vec, (17, 4));
ltx_lines = [join(row, " & ")*"\\\\" for row in eachrow(ltx_mat)];
println.(ltx_lines);
for i in gold_idxs
    println("\\draw[black,thick]([shift={(-0.3em,2.8ex)}]pic cs:start$i)rectangle([shift={(0.3em,-0.8ex)}]pic cs:end$i);")
end

## Print all reactions with bm
ltx_vec = [
    begin 
        s = string(rx_vec[rx])
        s = s[findfirst(' ', s)+1:end]
        s = Base.replace(s, "-->" => (
            rx ∈ gold_idxs ? 
            "} & \\bm{\\xrightarrow{k_{$rx}}" :
            "&\\xrightarrow{k_{$rx}}"
        ))
        s = Base.replace(s, "X" => "X_")
        s = Base.replace(s, "*" => "")
        rx ∈ gold_idxs ? "\\bm{$s}" : s
    end for rx in 1:n_rx
];
for i in [68]
    insert!(ltx_vec, i, " & ")
end
ltx_mat = reshape(ltx_vec, (17, 4));
ltx_lines = [join(row, " & ")*"\\\\" for row in eachrow(ltx_mat)];
println.(ltx_lines);

# Shared reactions
for pen_str in PEN_STRS
    println(pen_str)
    sort_by_w = sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
    for s in 5:5:25
        shared = reduce(intersect, first.(sort_by_w[1:s]))
        println("$s $shared")
    end
end

# Compare to gold
mat = zeros(Int64, (n_gold_rx, length(PEN_STRS)));
for (p, pen_str) in enumerate(PEN_STRS)
    sort_by_w = sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
    top10 = first.(sort_by_w[1:10])
    for (i, gold_idx) in enumerate(gold_idxs)
        mat[i,p] = sum(gold_idx .∈ top10)
    end
    display(mean(length.(top10)))
end
mat

### End results for paper





function plot_isol(isol, inferred)
    masked = zeros(n_rx);
    masked[inferred] .= isol.kvec[inferred];
    isol_remade = remake(isol.iprob.oprob, p=[k=>masked]);
    isol_osol = solve(isol_remade, saveat=t_grid);
    f = Figure()
    for i in 1:6
        ax_i = div(i-1, 3) + 1
        ax_j = mod(i-1, 3) + 1
        Axis(f[ax_i, ax_j])
        lines!(t_grid, getindex.(isol_osol.u, i))
        scatter!(t_obs, data[i,:])
    end
    f
end

pen_str = "logL1"

sort_by_b = sort(collect(bic_dict_lookup[pen_str]), by=(x)->x.second);
sort_by_w = sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
jac_dict = Dict(idxs => calc_jac(idxs, gold_idxs) for idxs in keys(post_dict_lookup[pen_str]));
sort_by_j = sort(collect(jac_dict), rev=true, by=(x)->x.second);

# Alt prior
posts = Dict(inferred => -0.5*bic+alt_logprior(length(inferred)) for (inferred, bic) in bic_dict_lookup[pen_str]);
logZ = logsumexp(collect(values(posts)));
alt_posts = Dict(inferred => exp(post-logZ) for (inferred, post) in posts);
sort_by_alt = sort(collect(alt_posts), rev=true, by=(x)->x.second);

# Largest log-likelihood
sort_by_p[1:10]
# Most similar to ground-truth CRN
sort_by_j[1:10]
# Highest probs
sort_by_w[1:10]
# Highest probs (unif)
sort_by_alt[1:10]
# Best BIC
sort_by_b[1:10]
# CRN complexity of highest prob
length.(first.(sort_by_w[1:10]))
# CRN complexity of highest prob (unif)
length.(first.(sort_by_alt[1:10]))
# Prob of CRNs most similar to ground truth
[post_dict_lookup[pen_str][idxs] for (idxs, _) in sort_by_j[1:10]]
# Highest probs
[post_dict_lookup[pen_str][idxs] for (idxs, _) in sort_by_w[1:10]]


pen_str = "logL1"
sort_by_w = sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
# Posterior mode
rk = 1
isol = isol_dict_lookup[pen_str][sort_by_w[rk].first];
isol.σs
isol.iprob.loss_func(isol.kvec, isol.σs)
sort_by_w[rk]
sort(isol.kvec ./ scale_fcts, rev=true)[1:20]
infer_reactions(isol, species_vec, rx_vec)[2]
plot_isol(isol, sort_by_w[rk].first)
(isol.kvec ./ scale_fcts)[sort_by_w[rk].first] # does not reach 1e2
sort_by_w[rk].first ∩ gold_idxs # recovered reactions

# Posterior rank 2
rk = 2
isol = isol_dict_lookup[pen_str][sort_by_w[rk].first];
isol.iprob.loss_func(isol.kvec, isol.σs)
infer_reactions(isol, species_vec, rx_vec)[2]
plot_isol(isol, sort_by_w[rk].first)
(isol.kvec ./ scale_fcts)[sort_by_w[rk].first] # does not reach 1e2
sort_by_w[rk].first ∩ gold_idxs # recovered reactions

# Top X
for pen_str in PEN_STRS
    println(pen_str)
    sort_by_w = sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
    # println.(sort_by_w[1:5])
    println(length.(first.(sort_by_w[1:5])))
    println(last.(sort_by_w[1:5]))
end

# Posterior mean of number of reactions
for pen_str in PEN_STRS
    mean_size = sum(length(rxs)*w for (rxs, w) in post_dict_lookup[pen_str])
    display(mean_size)
end

f = Figure(size=(900, 360));
for (i, pen_str) in enumerate(PEN_STRS)
    ax = Axis(
        f[1,i], title=L"$%$(pen_names[PEN_STRS[i]])$ \textbf{penalty}",
        xlabel="Number of CRNs", ylabel=i==1 ? "Average Jacaard similarity" : ""
    )
    sort_by_p = sort(collect(logp_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
    sort_by_w = sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
    jac_dict = Dict(idxs => calc_jac(idxs, gold_idxs) for idxs in keys(post_dict_lookup[pen_str]));
    n = 50;
    wtop_subset = first.(sort_by_w[1:n]);
    wtop_jacs = [jac_dict[idxs] for idxs in wtop_subset];
    jac_wmeans = cumsum(wtop_jacs) ./ (1:n);
    ptop_subset = first.(sort_by_p[1:n]);
    ptop_jacs = [jac_dict[idxs] for idxs in ptop_subset];
    jac_pmeans = cumsum(ptop_jacs) ./ (1:n);    
    lines!(jac_pmeans, label="Sorted by log-likelihood");
    lines!(jac_wmeans, label="Sorted by posterior probability");
    if i == length(PEN_STRS)
        f[2,:] = Legend(f, ax, orientation=:horizontal)
    end
end
f
save(joinpath(fig_dir, "jac_sim.png"), f);

# Number of correctly identified
f = Figure(size=(900, 360));
for (i, pen_str) in enumerate(PEN_STRS)
    ax = Axis(
        f[1,i], title=L"$%$(pen_names[PEN_STRS[i]])$ \textbf{penalty}",
        xlabel="Number of CRNs", ylabel=i==1 ? "Average Jacaard similarity" : ""
    )
    # sort_by_p = sort(collect(logp_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
    sort_by_w = sort(collect(post_dict_lookup[pen_str]), rev=true, by=(x)->x.second);
    # jac_dict = Dict(idxs => calc_jac(idxs, gold_idxs) for idxs in keys(post_dict_lookup[pen_str]));
    n = 25;
    wtop_subset = first.(sort_by_w[1:n]);
    lines!(1:n, length.(wtop_subset), label="Number of inferred reactions");
    lines!(1:n, length.(wtop_subset .∩ Ref(gold_idxs)), label="Number of inferred reactions in gold standard");
    if i == length(PEN_STRS)
        f[2,:] = Legend(f, ax, orientation=:horizontal)
    end
end
f



### TVD
function calc_tvd(wdict1, wdict2)
    all_idxs = keys(wdict1) ∪ keys(wdict2)
    return 0.5 * sum(abs(get(wdict1, idxs, 0.0)-get(wdict2, idxs, 0.0)) for idxs in all_idxs)
end

for p1 in 1:4
    for p2 in (p1+1):4
        pen1, pen2 = PEN_STRS[p1], PEN_STRS[p2]
        wdict1 = post_dict_lookup[pen1]
        wdict2 = post_dict_lookup[pen2]
        tvd = calc_tvd(wdict1, wdict2)
        println("$pen1, $pen2: $tvd")  
    end
end

for p1 in 1:4
    for p2 in (p1+1):4
        pen1, pen2 = PEN_STRS[p1], PEN_STRS[p2]
        wdict1 = post_dict_lookup[pen1]
        wdict2 = post_dict_lookup[pen2]
        shared = length(keys(wdict1) ∩ keys(wdict2))
        jac = calc_jac(keys(wdict1), keys(wdict2))
        println("$pen1, $pen2: $shared $jac")  
    end
end

length(reduce(∩, [keys(post_dict_lookup[p]) for p in PEN_STRS[2:4]]))
length(reduce(∪, [keys(post_dict_lookup[p]) for p in PEN_STRS[2:4]]))

reduce(∩, [keys(post_dict_lookup[p]) for p in PEN_STRS[2:4]])



bic_merged = Dict{Vector{Int64},Float64}();
for (pen_str, bdict) in bic_dict_lookup
    for (inferred, bic) in bdict
        if !haskey(bic_merged, inferred) || bic < bic_merged[inferred]
            bic_merged[inferred] = bic
        end
    end
end

unnormalised = Dict(rxs => -0.5*bic+logprior(length(rxs)) for (rxs, bic) in bic_merged);
logZ = logsumexp(collect(values(unnormalised)))
post_merged =  Dict(rxs => exp(logj-logZ) for (rxs, logj) in unnormalised);

for pen_str in PEN_STRS
    tvd = calc_tvd(post_dict_lookup[pen_str], post_merged)
    println("$pen_str: $tvd")
end

###




# Decompose posterior
sum(log.(isol_dict_lookup[pen_str][sort_by_w[1].first].σs ./ isol_dict_lookup[pen_str][sort_by_w[2].first].σs))

logp_dict_lookup[pen_str][sort_by_w[1].first]-logp_dict_lookup[pen_str][sort_by_w[2].first]
-0.5*log(length(data))*(length(sort_by_w[1].first)-length(sort_by_w[2].first))

-0.5*(bic_dict_lookup[pen_str][sort_by_w[1].first]-bic_dict_lookup[pen_str][sort_by_w[2].first])
logprior(length(sort_by_w[1].first))-logprior(length(sort_by_w[2].first))
log(sort_by_w[1][2] / sort_by_w[2][2])

calc_hess_eigvals(sort_by_w[1].first, pen_str)
calc_hess_eigvals(sort_by_w[2].first, pen_str)
@time calc_hess_eigvals(sort_by_w[3].first, pen_str)

-0.5*sum(log, calc_hess_eigvals(sort_by_w[1].first, pen_str))
-0.5*sum(log, calc_hess_eigvals(sort_by_w[2].first, pen_str))

(-0.5*sum(log, calc_hess_eigvals(sort_by_w[1].first, pen_str))
+0.5*sum(log, calc_hess_eigvals(sort_by_w[2].first, pen_str))
+0.5length(sort_by_w[1].first)*log(2π)-0.5length(sort_by_w[2].first)*log(2π))





# Try Hessian
using LinearAlgebra, ForwardDiff
tmp_infer = sort_by_w[4].first
isol = isol_dict_lookup[pen_str][tmp_infer];

isol.iprob.loss_func(isol.kvec, isol.σs)
isol.iprob.loss_func(mask_kvec(isol, tmp_infer), isol.σs)
infer_reactions(isol, species_vec, rx_vec)[2]

curr_init = [isol.est[tmp_infer];isol.est[n_rx+1:end]];
refine_func = make_refine_func(isol, tmp_infer; alg, abstol);
@time hess_init = ForwardDiff.hessian(refine_func, curr_init);
eigvals(hess_init)

function calc_hess_eigvals(idxs, pen_str)
    isol = isol_dict_lookup[pen_str][idxs];
    curr_init = [isol.est[idxs];isol.est[n_rx+1:end]];
    refine_func = make_refine_func(isol, idxs; alg, abstol);
    cfg = ForwardDiff.HessianConfig(refine_func, curr_init);
    hess_init = ForwardDiff.hessian(refine_func, curr_init, cfg);
    eigvals(hess_init)
end

# using BenchmarkTools
# AutoVern7(KenCarp4())
# AutoTsit5(Rosenbrock23()))

curr_init = [isol.est[tmp_infer];isol.est[n_rx+1:end]];

abstol = 1e-12;
refine_func = make_refine_func(isol, tmp_infer; alg=AutoVern7(KenCarp4()), abstol=abstol);
res = optimize(
    refine_func, 
    log.([fill(1e-10, length(tmp_infer)); fill(1e-5, n_species)]),
    log.([fill(1e2, length(tmp_infer)); fill(1e-1, n_species)]),
    curr_init,
    Fminbox(BFGS());
    autodiff = :forward
)
res_12 = res;
res_12.minimum
ForwardDiff.gradient(refine_func, res_12.minimizer)
hess_12 = ForwardDiff.hessian(refine_func, res_12.minimizer);
eigvals(hess_12)

abstol = 1e-14
curr_init = copy(res_12.minimizer);
refine_func = make_refine_func(isol, tmp_infer; alg=AutoVern7(KenCarp4()), abstol=abstol);
res_14 = optimize(
    refine_func, 
    log.([fill(1e-10, length(tmp_infer)); fill(1e-5, n_species)]),
    log.([fill(1e2, length(tmp_infer)); fill(1e-1, n_species)]),
    curr_init,
    Fminbox(BFGS());
    autodiff = :forward
)
res_14.minimum
ForwardDiff.gradient(refine_func, res_14.minimizer)
hess_14 = ForwardDiff.hessian(refine_func, res_14.minimizer);
eigvals(hess_14)

abstol = eps(Float64)
curr_init = copy(res_14.minimizer);
refine_func = make_refine_func(isol, tmp_infer; alg=AutoVern7(KenCarp4()), abstol=abstol);
res_eps = optimize(
    refine_func, 
    log.([fill(1e-10, length(tmp_infer)); fill(1e-5, n_species)]),
    log.([fill(1e2, length(tmp_infer)); fill(1e-1, n_species)]),
    curr_init,
    Fminbox(BFGS());
    autodiff = :forward
)
res_eps.minimum
ForwardDiff.gradient(refine_func, res_eps.minimizer)
hess_eps = ForwardDiff.hessian(refine_func, res_eps.minimizer);
eigvals(hess_eps)
sum(log(e) for e in eigvals(hess_eps) if e > 0)

# original starting pt
res_eps.minimum
ForwardDiff.gradient(refine_func, res_eps.minimizer)
hess_eps = ForwardDiff.hessian(refine_func, res_eps.minimizer);
eigvals(hess_eps)
sum(log(e) for e in eigvals(hess_eps) if e > 0)

# Before refining
curr_init = [isol.est[tmp_infer];isol.est[n_rx+1:end]];
hess_init = ForwardDiff.hessian(refine_func, curr_init);
eigvals(hess_init)
sum(log(e) for e in eigvals(hess_init) if e > 0)




# Do gold standard fit [x]
# Check scaling [x]
# Compare loss func [x]
# Different lambda [x]
# Check parsimonious CRNs [x]
# Different prior [x]

# [0, 2]?
# sort vectors in `rxs_no_k` [x]
# only export if optim success [x]
# change solver [x]

println(sort(gold_idxs))
for pen_str in PEN_STRS
    over_thres = [];
    for (inferred, bic) in bic_dict_lookup[pen_str]
        if -0.5*bic+logprior(length(inferred)) > 420
            push!(over_thres, inferred)
        end
    end
    println(reduce(∩, over_thres))
end