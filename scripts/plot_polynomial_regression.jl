include("src/linear_regression.jl")
include("src/therm_int.jl")

using ValueHistories
using BSON
using ColorSchemes
colors = ColorSchemes.seaborn_colorblind

# scale = sum(extrema(D.t))
# problem_params[:true_w] ./= scale
# D.t ./= scale
function run_experiment(problem_params, alg_params, D)
    initial_dist, q, hist = fit_linear_regression(problem_params, alg_params, D)

    H₀ = Distributions.entropy(initial_dist)
    EV = ( 
    true_gauss_expectation(initial_dist,  
                RegressionModel(problem_params[:ϕ], mean(initial_dist), 
                                problem_params[:true_β]), D)
            # num_expectation( initial_dist, 
            #         w -> log_likelihood(D, RegressionModel(problem_params[:ϕ], w, 
            #                                            problem_params[:true_β])) )
           + SVGD.expectation_V(initial_dist, initial_dist) 
           + 0.5 * logdet(2π * problem_params[:Σ_prior])
          )
    est_logZ_rkhs = SVGD.estimate_logZ(H₀, EV, KL_integral(hist)[end])

    true_logZ = regression_logZ(problem_params[:Σ_prior], problem_params[:true_β], 
                                problem_params[:true_ϕ], D.x)
    function logp(w)
        model = RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
        log_likelihood(D, model) + logpdf(MvNormal(problem_params[:μ_prior], problem_params[:Σ_prior]), w)
    end  
    function grad_logp(w) 
        model = RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
        (grad_log_likelihood(D, model) 
         .- inv(problem_params[:Σ_prior]) * (w-problem_params[:μ_prior])
        )
    end
    grad_logp!(g, w) = g .= grad_logp(w)

    # use eithe prior as initial distribution of change initial mean to MAP
    problem_params[:μ_prior] = if problem_params[:MAP_start]
        Optim.maximizer(Optim.maximize(logp, grad_logp!, problem_params[:μ_prior], LBFGS()))
        # posterior_mean(problem_params[:ϕ], problem_params[:true_β], D, 
        #                problem_params[:μ_prior], problem_params[:Σ_prior])
    else
        problem_params[:μ_prior]
    end

    DIRNAME = "polynomial_regression"
    file_prefix = savename( merge(problem_params, alg_params) )
    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
        merge(alg_params, problem_params, @dict(hist, q)),
                safe=true)
    return est_logZ_rkhs, true_logZ, hist, q
end

function plot_rkhs(data, plt, color; size=(275,275), legend=:bottomright, ylims=(-Inf,Inf), lw=3)
    dKL_hist = data[:hist]
    true_logZ = regression_logZ(problem_params[:Σ_prior], problem_params[:true_β], 
                                problem_params[:true_ϕ], D.x)
    initial_dist = MvNormal(problem_params[:μ_prior], problem_params[:Σ_prior])
    H₀ = Distributions.entropy(initial_dist)
    EV = ( 
          true_gauss_expectation(initial_dist,  
                RegressionModel(problem_params[:ϕ], mean(initial_dist), 
                                problem_params[:true_β]), D)
           + SVGD.expectation_V(initial_dist, initial_dist) 
           + 0.5 * logdet(2π * problem_params[:Σ_prior])
          )
    est_logZ = SVGD.estimate_logZ.(H₀, EV, KL_integral(data[:hist]))
    plot!(plt, est_logZ, label="$(data[:n_particles])", color=color, lw=lw, alpha=0.8);
    hline!(plt, [true_logZ], color=colors[8], label="", lw=lw);
end
function plot_multiple(data_set; legend, size)
    plt = plot(xlabel="iterations", ylabel="log Z", legend=legend, size=size);
    for (i, d) in enumerate(data_set)
        plot_rkhs(d, plt, colors[i])
    end
    return plt
end

# set up
n_dim = 3
problem_params = Dict(
    :n_samples => 20,
    :sample_range => [-3, 3],
    :true_ϕ => x -> [1, x, x^2],
    :true_w => [2, -1, 0.2],
    :true_β => 2,
    :ϕ => x -> [1, x, x^2],
    :μ_prior => zeros(n_dim),
    :Σ_prior => 0.1I(n_dim),
    :MAP_start => true,
)
true_model = RegressionModel(problem_params[:true_ϕ],
                             problem_params[:true_w], 
                             problem_params[:true_β])

D = generate_samples(model=true_model, n_samples=problem_params[:n_samples],
                     sample_range=problem_params[:sample_range])

alg_params = Dict(
    :step_size => 0.001,
    :n_iter => 1000,
    :n_particles => 2000,
    :kernel_width => "median_trick",
)
est_logZ_rkhs, true_logZ, hist = run_experiment(problem_params, alg_params, D)

true_logZ = regression_logZ(problem_params[:Σ_prior], problem_params[:true_β], 
problem_params[:true_ϕ], D.x)
initial_dist = MvNormal(problem_params[:μ_prior], problem_params[:Σ_prior])
H₀ = Distributions.entropy(initial_dist)
EV = ( 
true_gauss_expectation(initial_dist,  
RegressionModel(problem_params[:ϕ], mean(initial_dist), 
problem_params[:true_β]), D)
+ SVGD.expectation_V(initial_dist, initial_dist) 
+ 0.5 * logdet(2π * problem_params[:Σ_prior])
)
int_plot = plot(title="log Z");
plot!(int_plot, SVGD.estimate_logZ.([H₀], [EV], KL_integral(hist)), label="", color=colors[1], lw=3)
hline!(int_plot, [true_logZ], color=colors[2], label="", lw=3)
@info "Value comparison" true_logZ  est_logZ_rkhs 

ALG_PARAMS = Dict(
    :step_size => 0.001,
    :n_iter => 500,
    :n_particles => [50, 100, 200],
    :kernel_width => "median_trick",
)
for d in dict_list(ALG_PARAMS)
    run_experiment(problem_params, d, D)
end

data_set = [BSON.load(n) for n in readdir("data/polynomial_regression", join=true)]

root_dest = "/home/lew/Documents/BCCN_Master/Stein_labrot/SteinIntegration_AABI/plots/"
dest = root_dest*"regression/"
data_set = [d for d in data_set if d[:μ_prior] != [0.,0,0]]
sort!(data_set, by=d->d[:n_particles])
d=data_set[1]
savefig(
        plot_multiple(data_set, legend=:bottomright, size=(275,275)),
        dest* "$(d[:n_iter])iter_stepsize=$(d[:step_size])" *".png"
       )
