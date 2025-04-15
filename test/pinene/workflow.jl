using StableRNGs

using Format
FMT_3DP = ".3f";

include(joinpath(@__DIR__, "../../src/inference.jl")); # imports functions used for inference
include(joinpath(@__DIR__, "../../src/plot_helper.jl"));
include(joinpath(@__DIR__, "setup.jl"));

# Smooth data
smooth_resvec = smooth_data.(eachrow(data), Ref(t_obs));

# Visualise smoothed data
t_fine = range(t_span..., 500)
f = Figure();
ax = Axis(f[1,1]);
for i in 1:n_species    
    scatter!(t_obs, data[i,:], color=(palette[i], 0.6))
    lines!(t_fine, eval_spline(smooth_resvec[i]..., t_fine), color=palette[i])
end
current_figure()

init_σs = [estim_σ(datarow .- eval_spline(res..., t_obs)) for (datarow, res) in zip(eachrow(data), smooth_resvec)]
scale_fcts = get_scale_fcts(smooth_resvec, range(extrema(t_obs)..., 50), species_vec, rx_vec, k)

PEN_STR = "logL1";
hyp_val = 1.;
OPT_DIR = joinpath(@__DIR__, "output", PEN_STR, "hyp_$(hyp_val)"); # directory for storing optimisation results
mkpath(OPT_DIR); # create directory 

iprob = make_iprob(
    oprob, k, t_obs, data, PEN_STR, hyp_val; scale_fcts
);

rng = StableRNG(1);
init_vecs = [rand(rng, n_rx) for _ in 1:N_RUNS];

function optim_callback(i, res)
    time_str = pyfmt(FMT_3DP, res.time_run)
    σs = exp.(res.minimizer[end-n_species+1:end])
    σs_str = join(pyfmt.(Ref(FMT_3DP), σs), " ")
    println("Run $i: $(time_str) seconds, σs = [$σs_str])")
    flush(stdout)
end

# Perform multi-start optimisation
@time res_vec = optim_iprob(
    iprob, [lbs; fill(0.05, n_species)], [ubs; fill(5., n_species)], 
    init_vecs[1:8], init_σs; callback_func=optim_callback
);

res = argmin((x)->x.minimum, res_vec)

@code_warntype iprob.optim_func.f(res.minimizer)

iprob.itf(res.minimizer[1:n_rx])
# res.minimizer
exp.(res.minimizer[n_rx+1:end])
iprob.loss_func(iprob.itf(res.minimizer[1:n_rx]), exp.(res.minimizer[n_rx+1:end]))

est_mat = reduce(hcat, [res.minimizer for res in res_vec]);

# Export estimated reaction rates
export_estimates(res_vec, OPT_DIR);

# Read in reactions rates
est_mat = readdlm(joinpath(OPT_DIR, "estimates.txt"));

isol = make_isol(iprob, est_mat);
isol.kvec
isol.σs
inferred, fit = infer_reactions(isol, species_vec, rx_vec)

rx_vec[sort(inferred)]
isol.kvec[3] / sum(isol.kvec[[1,2,3,13]])
isol.σs



# Tsu?
tsu_rxs = [1, 2, 9, 15, 19]
tsu_kvec = zeros(n_rx);
tsu_kvec[tsu_rxs] = [2.13, 0.91, 0, 1.22, 0.89] ./ maximum(t_obs)
tsu_kvec[9] = 2.047e-5;
tsu_θ = [iprob.tf(tsu_kvec); fill(log(0.1), n_species)];

tsu_res = optim_iprob(
    iprob, [lbs; fill(0.05, n_species)], [ubs; fill(5., n_species)], 
    [tsu_kvec ./ iprob.scale_fcts], init_σs; callback_func=optim_callback
)[1]

tsu_isol = make_isol(iprob, reshape(tsu_res.minimizer, (n_rx+n_species,1)));
tsu_inferred, tsu_fit = infer_reactions(tsu_isol, species_vec, rx_vec);

tsu_isol.σs