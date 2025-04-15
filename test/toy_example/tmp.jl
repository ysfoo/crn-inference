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

init_σs_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
scale_dict = Dict{Tuple{Float64,Float64}, Vector{Float64}}();
for k1 in K1_VALS, k18 in K18_VALS
    data_dir = get_data_dir(k1, k18)
    t_obs, data = read_data(joinpath(data_dir, "data.txt"));
	smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));
    init_σs_dict[(k1, k18)] = [
		estim_σ(datarow .- eval_spline(res..., t_obs)) 
		for (datarow, res) in zip(eachrow(data), smooth_resvec)
	]
	scale_dict[(k1, k18)] = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)
end

## Choose best run for each dataset + penalty func
logprior(n) = -logabsbinomial(n_rx, n)[1] - log(n_rx+1)

tmp_crn = [1, 5, 13, 18, 21]
k1=0.1; k18=10.0; pen_str="logL1"

refine_func_dict = Dict{Vector{Int64}, FunctionWrapper}();
data_dir = get_data_dir(k1, k18);
t_obs, data = read_data(joinpath(data_dir, "data.txt"));
scale_fcts = scale_dict[(k1, k18)];
σ_lbs = 0.01.*init_σs_dict[(k1, k18)];
σ_ubs = std.(eachrow(data))

bic_by_rxs = Dict{Vector{Int64},Float64}();
isol_by_rxs = Dict{Vector{Int64},ODEInferenceSol}();
iprob = make_iprob(
    oprob, k, t_obs, data, pen_str, HYP_VALS[1]; 
    scale_fcts, abstol=1e-10,
);
for hyp_val in HYP_VALS
    opt_dir = get_opt_dir(k1, k18, pen_str, hyp_val);  
    # pen_func = make_pen_func(pen_str, hyp_val)           
    est_mat = readdlm(joinpath(opt_dir, "estimates.txt"));
    # n_species = size(iprob.data, 1)
    for est in eachcol(est_mat)
        kvec = iprob.itf(est[1:end-n_species])
        σs = exp.(est[end-n_species+1:end])
        isol = ODEInferenceSol(iprob, est, kvec, σs)
        inferred, fit = infer_reactions(isol, species_vec, rx_vec)
        inferred = sort(inferred)
        bic = 2*fit + length(inferred)*log(length(data))
        if inferred == [1,13,18]
            display((hyp_val, median(est)))
        end
        if !haskey(bic_by_rxs, inferred) || bic < bic_by_rxs[inferred]
            bic_by_rxs[inferred] = bic
            isol_by_rxs[inferred] = isol
        end
    end
end
logp_by_rxs = sort(
    collect(Dict(rxs => -0.5*bic+logprior(length(rxs)) for (rxs, bic) in bic_by_rxs)),
    by=x->x.second, rev=true
)
max_logp = maximum(last.(logp_by_rxs))
to_refine = [crn for (crn, logp) in logp_by_rxs if max_logp - logp < log(1e4)]
# display((length(to_refine), length(logp_by_rxs)))    
# to_refine = first.(sort(collect(logp_by_rxs), by=x->x.second, rev=true)[1:25])
# display(max_logp - logp_by_rxs[25].second)

bic_by_rxs[tmp_crn]
minimum(values(bic_by_rxs))

for crn in to_refine
    if !haskey(refine_func_dict, crn)
        iprob = make_iprob(
            oprob, k, t_obs, data, pen_str, HYP_VALS[1]; 
            scale_fcts, abstol=1e-10, alg=AutoVern7(KenCarp4()), verbose=false
        );
        refine_func_dict[crn] = make_refine_func(iprob, crn, σ_lbs, σ_ubs)
    end
end

isol = isol_by_rxs[tmp_crn];
refine_func = refine_func_dict[tmp_crn];
isol.iprob.loss_func(isol.kvec, isol.σs)
isol.iprob.loss_func(mask_kvec(isol, tmp_crn), isol.σs)
isol.kvec

refined_isol, refine_res, refine_func = refine_isol(
    refine_func, isol, tmp_crn, 
    σ_lbs, σ_ubs;
    optim_opts=Optim.Options(iterations=1, time_limit=600.)
);

refined_isol, refine_res, refine_func = refine_isol(
    refine_func, isol, tmp_crn, 
    σ_lbs, σ_ubs;
    optim_alg = BFGS(linesearch=LineSearches.BackTracking()),
    optim_opts=Optim.Options(iterations=10^4, time_limit=600.)
);


dict_lock = ReentrantLock()
enumerate_crns = collect(enumerate(to_refine))
@time @sync for (i, crn) in enumerate_crns
    Threads.@spawn begin
        isol = @lock dict_lock isol_by_rxs[crn]
        refine_func = refine_func_dict[crn]
        # println("$i")
        # flush(stdout)
        refined_isol, refine_res, refine_func = try
            refine_isol(
                refine_func, isol, crn, 
                σ_lbs, σ_ubs;
                optim_opts=Optim.Options(iterations=10^4, time_limit=600.)
            )
        catch e
            println("HagerZhang failed for $crn, k1=$k1, k18=$k18, pen=$pen_str")
            refine_isol(
                refine_func, isol, crn, 
                σ_lbs, σ_ubs;
                optim_alg = BFGS(linesearch=LineSearches.BackTracking()),
                optim_opts=Optim.Options(iterations=10^4, time_limit=600.)
            )
        end
        @lock dict_lock isol_by_rxs[crn] = refined_isol
        # println("$i $(refine_res.iterations) $(refine_res.f_calls) $(refine_res.g_calls) $(refine_res.time_run)")
        # refined_bic = 2*refine_res.minimum + length(crn)*log(length(data))
        # println("$(round(bic_by_rxs[crn];digits=4)) $(round(refined_bic;digits=4))")
        # flush(stdout)
    end
end