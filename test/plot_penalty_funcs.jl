################################################################################################
### Define ground truth reaction network and the reaction network library used for inference ###
################################################################################################

SRC_DIR = joinpath(@__DIR__, "../src");
include(joinpath(SRC_DIR, "inference.jl")); # imports penalty functions
include(joinpath(SRC_DIR, "plot_helper.jl"));

xs = 10 .^ range(-10, 2, 1001);
xmax = 1
p_trg = hslike_pen(xmax, 1) - hslike_pen(1e-10, 1)
L1_hyp = p_trg / xmax
logL1_hyp = p_trg / (log(xmax) + 10log(10))
approxL0_hyp = p_trg / (xmax^0.1 - 0.1)

L1_vals = L1_pen.(xs, L1_hyp) .- L1_pen(1e-10, L1_hyp);
logL1_vals = logL1_pen.(xs, logL1_hyp) .- logL1_pen(1e-10, logL1_hyp);
approxL0_vals = approxL0_pen.(xs, approxL0_hyp) .- approxL0_pen(1e-10, approxL0_hyp);
hslike_vals = hslike_pen.(xs, 1.0) .- hslike_pen(1e-10, 1.0);

f = begin
f = Figure(size=(400,300));
ax = Axis(
    f[1,1], xlabel=L"Rate constant $k$", title=L"\text{pen}(k) - \text{pen}(10^{-10})",
    xscale=log10, xticks=LogTicks(LinearTicks(6)), yticks=[0]
);
lines!(xs, L1_vals, label=L"$L_1$ (lasso)", linestyle=:solid);
lines!(xs, logL1_vals, label=L"Log scale $L_1$", linestyle=(:dash, :dense));
lines!(xs, approxL0_vals, label=L"Approximate $L_0$", linestyle=:dot);
lines!(xs, hslike_vals, label="Horseshoe-like", linestyle=(:dashdot, :dense));
xlims!(1e-10, 1e2);
ylims!(-0.2, 12);
axislegend(position=:lt);
f
end
save(joinpath(@__DIR__, "penalty_funcs.png"), f)