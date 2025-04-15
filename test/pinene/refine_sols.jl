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
include(joinpath(@__DIR__, "setup.jl"));

smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
scale_fcts = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k);

## Choose best run for each dataset + penalty func
logprior(n) = -logabsbinomial(n_rx, n)[1] - log(n_rx+1)

refine_func_dict = Dict{Vector{Int64}, FunctionWrapper}();

for pen_str in PEN_STRS
    bic_by_rxs = Dict{Vector{Int64},Float64}();
    isol_by_rxs = Dict{Vector{Int64},ODEInferenceSol}();
    for hyp_val in HYP_VALS
        opt_dir = get_opt_dir(pen_str, hyp_val);        
        # pen_func = make_pen_func(pen_str, hyp_val)
        iprob = make_iprob(
            oprob, k, t_obs, data, pen_str, hyp_val; 
            scale_fcts, abstol=1e-10,
            # custom_pen_func=(x)->pen_func(sum(x[13:16])*sum(x[21:24]))
        );
        
        est_mat = readdlm(joinpath(opt_dir, "estimates.txt"));
        # n_species = size(iprob.data, 1)
        for est in eachcol(est_mat)
            kvec = iprob.itf(est[1:end-n_species])
            σs = exp.(est[end-n_species+1:end])
            isol = ODEInferenceSol(iprob, est, kvec, σs)
            bic_dict = map_isol(isol, species_vec, rx_vec; abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false, thres=10.)
            for (crn, bic) in bic_dict
                if !haskey(bic_by_rxs, crn) || bic < bic_by_rxs[crn]
                    bic_by_rxs[crn] = bic
                    isol_by_rxs[crn] = isol
                end
            end
        end
    end
    logp_by_rxs = sort(
        collect(Dict(rxs => -0.5*bic+logprior(length(rxs)) for (rxs, bic) in bic_by_rxs)),
        by=x->x.second, rev=true
    )
    max_logp = maximum(last.(logp_by_rxs))
    to_refine = [crn for (crn, logp) in logp_by_rxs if max_logp - logp < log(1e4)]
    display((length(to_refine), length(logp_by_rxs)))    
    for crn in to_refine
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
    @time @sync for (i, crn) in enumerate_crns
        Threads.@spawn begin
            # display(crn)
            isol = @lock dict_lock isol_by_rxs[crn]
            refine_func = refine_func_dict[crn]
            refined_isol, refine_res, refine_func = try
                refine_isol(
                    refine_func, isol, crn, 
                    σ_lbs, σ_ubs;
                    optim_opts=Optim.Options(iterations=10^4, time_limit=600.)
                )
            catch e
                println("HagerZhang failed for $crn, pen=$pen_str")
                refine_isol(
                    refine_func, isol, crn, 
                    σ_lbs, σ_ubs;
                    optim_alg = BFGS(linesearch=LineSearches.BackTracking()),
                    optim_opts=Optim.Options(iterations=10^4, time_limit=600.)
                )
            end
            @lock dict_lock isol_by_rxs[crn] = refined_isol
            println("$i $(refine_res.iterations) $(refine_res.f_calls) $(refine_res.g_calls) $(refine_res.time_run)")
            # refined_bic = 2*refine_res.minimum + length(crn)*log(length(data))
            # println("$(round(bic_by_rxs[crn];digits=4)) $(round(refined_bic;digits=4))")
            flush(stdout)
        end
    end
    est_mat = reduce(hcat, [isol.est for isol in values(isol_by_rxs)])
    fname = joinpath(@__DIR__, "output", pen_str, "refined_estimates.txt")
    writedlm(fname, est_mat);
end