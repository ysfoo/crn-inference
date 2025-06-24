#####################################################################
### Make summary plots for best run of each optimisation instance ###
#####################################################################

using ProgressMeter
using LogExpFunctions, SpecialFunctions
using StatsBase

using StableRNGs

using LinearAlgebra
BLAS.set_num_threads(1);

### Try to make multi-threading more efficient
using ThreadPinning
isslurmjob() = get(ENV, "SLURM_JOBID", "") != ""
isslurmjob() ? pinthreads(:affinitymask) : pinthreads(:cores);
###

include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference
include(joinpath(@__DIR__, "setup.jl"));

MAX_DIST = 2; # 0 for no crossover
MAX_DIFFS = 1000;
EST_FNAME = (MAX_DIST == 0) ? "nocross_estimates.txt" : "refined_estimates.txt"

smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
scale_fcts = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k);

## Choose best run for each dataset + penalty func
logprior(n) = -logabsbinomial(n_rx, n)[1] - log(n_rx+1)

refine_func_dict = Dict{Vector{Int64}, FunctionWrapper}();

for pen_str in PEN_STRS
    bic_by_rxs = Dict{Vector{Int64},Float64}();
    isol_by_rxs = Dict{Vector{Int64},ODEInferenceSol}();
    iprob = make_iprob(
        oprob, k, t_obs, data, pen_str, HYP_VALS[1]; 
        scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
    );
    estim_σs_func = make_estim_σs_func(iprob.oprob, iprob.k, iprob.t_obs, iprob.data; abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false)
    @time for hyp_val in HYP_VALS
        opt_dir = get_opt_dir(pen_str, hyp_val);       
        est_mat = readdlm(joinpath(opt_dir, "estimates.txt"));
        for est in eachcol(est_mat)
            kvec = iprob.itf(est[1:n_rx])
            σs = exp.(est[n_rx+1:end])
            isol = ODEInferenceSol(iprob, est, kvec, σs)
            bic_dict = map_isol(
                isol, species_vec, rx_vec;
                estim_σs_func=estim_σs_func,
                abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false, thres=log(1e6)
            )
            for (crn, bic) in bic_dict
                crn = sort(crn)
                mask_est = copy(est)
                if !haskey(bic_by_rxs, crn) || bic < bic_by_rxs[crn]
                    mask_est[1:n_rx] .= -Inf
                    mask_est[crn] .= est[crn]
                    bic_by_rxs[crn] = bic
                    isol_by_rxs[crn] = ODEInferenceSol(iprob, mask_est, iprob.itf(mask_est[1:n_rx]), σs)
                end
            end
        end
    end
    flush(stdout)
    
    orig_sort_by_logp = sort(
        collect(Dict(rxs => -0.5*bic+logprior(length(rxs)) for (rxs, bic) in bic_by_rxs)), rev=true,
        by=last
    )
    max_logp = orig_sort_by_logp[1].second
    thres = max(log(1e6), max_logp-orig_sort_by_logp[25].second)
    n_crns = length(isol_by_rxs)
    top_crns = [crn for (crn, logp) in orig_sort_by_logp if max_logp - logp <= thres];
    n_top = length(top_crns)
    max_crn_size = maximum(length.(top_crns))

    cross_seeds = [crn for (crn, _) in orig_sort_by_logp[1:min(1000, n_crns)]];
    all_diffs = Dict{Pair{Vector{Int64},Vector{Int64}}, Vector{Tuple{Float64,Vector{Int64}}}}(); # map (alt reactions, replaced reactions) to vector of (loss diff, CRN with alt reactions)
    for crn1 in cross_seeds, crn2 in cross_seeds
        crn_diff = find_diff_pair(crn1, crn2)
        if (length(crn_diff.first) <= MAX_DIST && length(crn_diff.second) <= MAX_DIST)
            rep_est = replace_est(isol_by_rxs[crn2].est, isol_by_rxs[crn1].est, crn_diff.first, crn_diff.second)
            rep_kvec = iprob.itf(rep_est[1:n_rx])
            est_σs = estim_σs_func(rep_kvec)
            if !all(isfinite.(est_σs)) continue end
            rep_σs = strict_clamp.(est_σs, σ_lbs, σ_ubs)
            rep_fit = iprob.loss_func(rep_kvec, rep_σs)
            base_fit = 0.5*(bic_by_rxs[crn2] - log(length(data))*length(crn2))
            haskey(all_diffs, crn_diff) || begin all_diffs[crn_diff] = [] end
            push!(all_diffs[crn_diff], (rep_fit - base_fit, crn1))
        end
    end
    sort_diffs = sort(collect(all_diffs), by=x->minimum(first.(x.second)))
    n_diffs = max(min(MAX_DIFFS, length(all_diffs)), findlast((x)->length(x.first.first)+length(x.first.second)==0, sort_diffs));
    crn_diffs = Dict(crn_diff => last.(alt_vec) for (crn_diff, alt_vec) in sort_diffs[1:n_diffs]);

    cand_ests = Dict{Vector{Int64}, Tuple{Float64,Vector{Float64}}}(); # map CRN to (loss, estimate)
    @time for crn in top_crns
        isol = isol_by_rxs[crn]
        for (crn_diff, alt_crns) in crn_diffs
            crn_diff.second ⊆ crn || continue # CRN needs to include all reactions to be replaced
            cand_crn = translate_crn(crn, crn_diff)
            # if length(cand_crn) > max_crn_size continue end
            for alt_crn in alt_crns
                rep_est = replace_est(isol.est, isol_by_rxs[alt_crn].est, crn_diff.first, crn_diff.second)
                rep_kvec = iprob.itf(rep_est[1:n_rx])
                est_σs = estim_σs_func(rep_kvec)
                if !all(isfinite.(est_σs)) continue end
                rep_σs = strict_clamp.(est_σs, σ_lbs, σ_ubs)
                rep_est[n_rx+1:end] .= log.(rep_σs)
                fit = iprob.loss_func(rep_kvec, rep_σs)
                if !haskey(cand_ests, cand_crn) || fit < cand_ests[cand_crn][1]
                    cand_ests[cand_crn] = (fit, rep_est)
                end
                length(crn_diff.first) == 0 && break # no reactions to be added, no need to loop through alt_crns
            end
        end
    end    
    cand_bics = Dict(crn => 2*fit+log(length(data))*length(crn) for (crn, (fit, est)) in cand_ests);
    cand_logps = Dict(crn => -fit-0.5*log(length(data))*length(crn)+logprior(length(crn)) for (crn, (fit, est)) in cand_ests);
    sort_by_logp = sort(collect(cand_logps), by=last, rev=true);
    n_cands = length(cand_logps);
    n_refine = min(n_cands, 1000)#clamp(findall(x->max_logp-x.second<log(1e10), sort_by_logp)[end], min(25, n_crns), min(1000, n_crns));  
    to_refine = [(crn, cand_ests[crn][2]) for (crn, logp) in sort_by_logp[1:n_refine]];
    display((pen_str, n_crns, n_top, max_crn_size, length(all_diffs), n_cands, n_refine));
    flush(stdout)
    
    for (crn, est) in to_refine
        if !haskey(refine_func_dict, crn)
            iprob = make_iprob(
                oprob, k, t_obs, data, pen_str, HYP_VALS[1]; 
                scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
            );
            refine_func_dict[crn] = make_refine_func(iprob, crn, σ_lbs, σ_ubs)
        end
    end        

    dict_lock = ReentrantLock()
    enumerate_crns = collect(enumerate(to_refine))
    @time @sync for (i, (crn, est)) in enumerate_crns
        Threads.@spawn begin
            # display(crn)
            refine_func = refine_func_dict[crn]
            refined_isol, refine_res = try
                refine_isol(
                    refine_func, iprob, est, crn, 
                    σ_lbs, σ_ubs;
                    optim_opts=Optim.Options(iterations=10^4, time_limit=600.)
                )
            catch e
                println("HagerZhang failed for $crn, pen=$pen_str")
                refine_isol(
                    refine_func, iprob, est, crn, 
                    σ_lbs, σ_ubs;
                    optim_alg = BFGS(linesearch=LineSearches.BackTracking()),
                    optim_opts=Optim.Options(iterations=10^4, time_limit=600.)
                )
            end
            @lock dict_lock isol_by_rxs[crn] = refined_isol
            # println("$i $(refine_res.iterations) $(refine_res.f_calls) $(refine_res.g_calls) $(refine_res.time_run)")
            # refined_bic = 2*refine_res.minimum + length(crn)*log(length(data))
            # println("$(round(cand_logps[crn];digits=4)) $(round(-iprob.loss_func(isol.kvec, isol.σs)-0.5*log(length(data))*length(crn)+logprior(length(crn));digits=4))")
            # flush(stdout)
        end
    end
    refined_logp = sort(
        [crn => -iprob.loss_func(isol.kvec, isol.σs)-0.5*log(length(data))*length(crn)+logprior(length(crn))
        for (crn, isol) in isol_by_rxs], by=last, rev=true
    )
    display(refined_logp[1:10])
    flush(stdout)
    est_mat = reduce(hcat, [isol.est for isol in values(isol_by_rxs)])
    fname = joinpath(@__DIR__, "output", pen_str, EST_FNAME)
    writedlm(fname, est_mat);
end