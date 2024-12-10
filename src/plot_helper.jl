#####################################
### Helper functions for plotting ###
#####################################

using CairoMakie
using Colors
using ColorSchemes

using Format
FMT_2DP = ".2f" # `pyfmt(FMT_2DP, num)` converts a float `num` to a string with 2 decimal points


# Plotting setup
set_theme!(theme_latexfonts());
update_theme!(
	Axis=(;
		xgridvisible=false, ygridvisible=false,
		xlabelsize=16, ylabelsize=16,
		titlesize=16,
	),
)
palette = Makie.wong_colors();

darken(c, w) = RGB(c.r*w, c.g*w, c.b*w);
lighten(c, w) = RGB(1-(1-c.r)*w, 1-(1-c.g)*w, 1-(1-c.b)*w);


# Make square root scale work for negative values
function symsqrt(x)
	sign(x)*sqrt(abs(x))
end

function Makie.inverse_transform(::typeof(symsqrt))
    x -> sign(x)*x*x
end

Makie.defaultlimits(::typeof(symsqrt)) = (0.0, 1.0)

Makie.defined_interval(::typeof(symsqrt)) = Makie.OpenInterval(-Inf, Inf)

function get_pos_sqrt_ticks(maxval)
	all_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100.];
	upper_filter = all_ticks .<= maxval
	lower_filter = all_ticks .>= 0.02*maxval
	return [0.0; all_ticks[upper_filter .& lower_filter]]
end