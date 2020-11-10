using DrWatson
using Plots
using BSON
using Distributions
using DataFrames
using LinearAlgebra
using Optim

using SVGD
# include(joinpath(@__DIR__, "src", "SVGD.jl"))

global DIRNAME = "linear_regression"

### local util functions

module Data
    struct RegressionModel
        ϕ  # feature maps
        w  # coefficients of feature maps
        β  # noise precision
    end
    struct RegressionData
        x  # input
        t  # target
    end
end
RegressionModel = Data.RegressionModel
RegressionData = Data.RegressionData

Base.getindex(d::RegressionData, i::Int) = RegressionData(d.x[i], d.t[i])
Base.display(d::RegressionData) = display([d.x d.t])
Base.length(d::RegressionData) = length(d.t)
Base.iterate(d::RegressionData) = (d[1], 1)
function Base.iterate(d::RegressionData, state)
    if state < length(d)
        (d[state+1], state+1)
    else
        return nothing
    end
end
y(model::RegressionModel) = x -> dot(model.w, model.ϕ(x))
y(model::RegressionModel, x) = y(model)(x)

# util functions for analytical solution
# returns an array (indexed by x) of arrays containing ϕ(x)
Φ(ϕ, X) = vcat( ϕ.(X)'... )
Φ(m::RegressionModel, X) = Φ(m.ϕ, X) 
# accuracy = inverse of variance
function posterior_accuracy(ϕ, β, X, Σ₀)
    inv(Σ₀) + β * Φ(ϕ, X)'Φ(ϕ, X)
end
function posterior_variance(ϕ, β, X, Σ₀)
    inv(posterior_accuracy(ϕ, β, X, Σ₀))
end
function posterior_mean(ϕ, β, D, μ₀, Σ₀)
    posterior_variance(ϕ, β, D.x, Σ₀) * ( inv(Σ₀)μ₀ + β * Φ(ϕ, D.x)' * D.t )
end
function regression_logZ(Σ₀, β, ϕ, X)
    2 \ log( det( 2π * posterior_variance(ϕ, β, X, Σ₀) ) ) 
end

function true_gauss_expectation(d::MvNormal, m::RegressionModel, D::RegressionData)
    X = reduce(hcat, m.ϕ.(D.x))
    @show size(X)
    0.5 * m.β * (tr((mean(d) * mean(d)' + cov(d)) * X * X')
        - 2 * D.t' * X' * mean(d)
        + D.t' * D.t
        + length(D) * log(m.β / 2π))
end

function generate_samples(;model::RegressionModel, n_samples=100, 
                          sample_range=[-10, 10])
    samples =  rand(Uniform(sample_range...), n_samples) 
    noise = rand(Normal(0, 1/sqrt(model.β)), n_samples)
    target = y(model).(samples) .+ noise
    return RegressionData(samples, target)
end

function likelihood(D::RegressionData, model::RegressionModel)
    prod( D-> pdf( Normal(y(model, D.x), 1/sqrt(model.β)), D.t), D )
end

E(D, model) = 2 \ sum( (D.t .- y(model).(D.x)).^2 )
function log_likelihood(D::RegressionData, model::RegressionModel)
    length(D)/2 * log(model.β/2π) - model.β * E(D, model)
end

function grad_log_likelihood(D::RegressionData, model::RegressionModel) 
    model.β * sum( ( D.t .- y(model).(D.x) ) .* model.ϕ.(D.x) )
end

function fit_linear_regression(problem_params, alg_params, D::RegressionData)
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
    global μ_prior = if problem_params[:MAP_start]
        Optim.maximizer(Optim.maximize(logp, grad_logp!, problem_params[:μ_prior], LBFGS()))
        # posterior_mean(problem_params[:ϕ], problem_params[:true_β], D, 
        #                problem_params[:μ_prior], problem_params[:Σ_prior])
    else
        problem_params[:μ_prior]
    end

    initial_dist = MvNormal(μ_prior, problem_params[:Σ_prior])
    q = rand(initial_dist, alg_params[:n_particles])

    q, hist = SVGD.svgd_fit(q, grad_logp; alg_params...)
    return initial_dist, q, hist
end
# the other numerical_expectation function applies f to each element instead
# of each col :/
function num_expectation(d::Distribution, f; n_samples=10000)
    sum( f, eachcol(rand(d, n_samples)) ) / n_samples
end

function run_linear_regression(problem_params, alg_params, n_runs)
    svgd_results = []
    estimation_rkhs = []
    estimation_unbiased = []
    estimation_stein_discrep = []

    true_model = RegressionModel(problem_params[:true_ϕ],
                                 problem_params[:true_w], 
                                 problem_params[:true_β])
    # dataset with labels
    D = generate_samples(model=true_model, 
                         n_samples=problem_params[:n_samples],
                         sample_range=problem_params[:sample_range]
                        )

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        initial_dist, q, hist = fit_linear_regression(problem_params, 
                                                      alg_params, D)
        H₀ = Distributions.entropy(initial_dist)
        EV = ( num_expectation( 
                    initial_dist, 
                    w -> log_likelihood(D, 
                            RegressionModel(problem_params[:ϕ], w, 
                                            problem_params[:true_β])) 
               )
               + expectation_V(initial_dist, initial_dist) 
               + 0.5 * log( det(2π * problem_params[:Σ_prior]) )
              )
        est_logZ_rkhs = estimate_logZ(H₀, EV,
                        alg_params[:step_size] * sum( get(hist,:dKL_rkhs)[2] ) 
                                 )
        est_logZ_unbiased = estimate_logZ(H₀, EV,
                    alg_params[:step_size] * sum( get(hist,:dKL_unbiased)[2] ) 
                                 )
        est_logZ_stein_discrep = estimate_logZ(H₀, EV,
                alg_params[:step_size] * sum( get(hist,:dKL_stein_discrep)[2] ) 
                                 )

        push!(svgd_results, (hist, q))
        push!(estimation_rkhs, est_logZ_rkhs) 
        push!(estimation_unbiased, est_logZ_unbiased)
        push!(estimation_stein_discrep,est_logZ_stein_discrep)
        @info est_logZ_rkhs
        @info est_logZ_unbiased
        @info est_logZ_stein_discrep
    end

    true_logZ = regression_logZ(problem_params[:Σ_prior], true_model.β, true_model.ϕ, D)
    @info true_logZ

    file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                @dict n_runs, true_logZ, estimation_unbiased, 
                        estimation_stein_discrep,
                        estimation_rkhs, svgd_results),
            safe=true, storepatch = false)
end

function plot_results(plt, q, problem_params)
    x = range(problem_params[:sample_range]..., length=100)
    for w in eachcol(q)
        model = RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
        plot!(plt,x, y(model), alpha=0.3, color=:orange, legend=:none)
    end
    plot!(plt,x, 
            y(RegressionModel(problem_params[:ϕ], 
                              mean(q, dims=2), 
                              problem_params[:true_β])), 
        color=:red)
    plot!(plt,x, 
            y(RegressionModel(problem_params[:true_ϕ], 
                              problem_params[:true_w], 
                              problem_params[:true_β])), 
        color=:green)
    return plt
end

## Experiments - linear regression on 3 basis functions

# alg_params = Dict(
#     :step_size => 0.0001,
#     :n_iter => 1000,
#     :n_particles => 20,
#     :kernel_width => "median_trick",
# )

# problem_params = Dict(
#     :n_samples => 20,
#     :sample_range => [-3, 3],
#     :true_ϕ => x -> [x, x^2, x^4, x^5],
#     :true_w => [2, -1, 0.2, 1],
#     :true_β => 2,
#     :ϕ => x -> [x, x^2, x^4, x^3],
#     :μ_prior => zeros(4),
#     :Σ_prior => 1.0I(4),
#     :MAP_start => true,
# )

# n_runs = 1

# run_linear_regression(problem_params, alg_params, n_runs)

# # run it once to get a value for log Z
# true_model = RegressionModel(problem_params[:true_ϕ], problem_params[:true_w], 
#                                  problem_params[:true_β])
#     # dataset with labels
# D = generate_samples(model=true_model, 
#                      n_samples=problem_params[:n_samples],
#                      sample_range=problem_params[:sample_range]
#                     )

# initial_dist, q, hist = fit_linear_regression(problem_params, alg_params, D)
# H₀ = Distributions.entropy(initial_dist)
# EV = ( num_expectation( 
#                     initial_dist, 
#                     w -> log_likelihood(D, 
#                             RegressionModel(problem_params[:ϕ], w, 
#                                             problem_params[:true_β])) 
#                )
#                + SVGD.expectation_V(initial_dist, initial_dist) 
#                + 0.5 * log( det(2π * problem_params[:Σ_prior]) )
#               )
# est_logZ = SVGD.estimate_logZ(H₀, EV,
#                             alg_params[:step_size] * sum( get(hist,:dKL_rkhs)[2] ) 
#                                  )

# norm_plot = plot(hist[:ϕ_norm], title = "φ norm", yaxis = :log)
# int_plot = plot(
#     SVGD.estimate_logZ.([H₀], [EV], alg_params[:step_size] * cumsum( get(hist, :dKL_rkhs)[2]))
#     ,title = "log Z", label = "",
# )
# fit_plot = plot_results(plot(size=(300,250)), q, problem_params)
# plot(fit_plot, norm_plot, int_plot, layout=@layout [f; n i])
