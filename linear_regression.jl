using DrWatson
using BSON
using Distributions
using DataFrames
using LinearAlgebra

using SVGD

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
Base.length(d::RegressionData) = length(d.x)
Base.iterate(d::RegressionData) = (d[1], 1)
function Base.iterate(d::RegressionData, state)
    if state < length(d)
        (d[state+1], state+1)
    else
        return nothing
    end
end
y(model::RegressionModel, x) = dot(model.w, model.ϕ(x))

function generate_samples(;model::RegressionModel, n_samples=100, 
                          sample_range=[-10, 10])
    samples =  rand(Uniform(sample_range...), n_samples) 
    noise = rand(Normal(0, 1/sqrt(model.β)), n_samples)
    target = y.([model], samples) .+ noise
    return RegressionData(samples, target)
end

function likelihood(data::RegressionData, model::RegressionModel)
    prod( data-> pdf( Normal(y(model, data.x), 1/sqrt(model.β)), data.t), data )
end

E(D, model) = 2 \ sum( (D.t .- y.([model], D.x)).^2 )
function log_likelihood(data::RegressionData, model::RegressionModel)
    length(data)/2 * log(model.β/(2π)) - model.β * E(data, model)
end

# function grad_logp(D::RegressionData, model::RegressionModel) 
#     model.β * sum( ( D.t .- y.([model], D.x) ) .* model.ϕ.(D.x) )
# end

function fit_linear_regression(problem_params, alg_params, D) 
    # we use the prior as the initial distribution of the particles
    initial_dist = MvNormal(problem_params[:μ_prior],
                            problem_params[:Σ_prior])

    q = rand(initial_dist, alg_params[:n_particles])
    function grad_logp(w) 
        model = RegressionModel(problem_params[:ϕ], w, problem_params[:true_β])
        # grad_log_prior = 0 #TODO add prior
        model.β * sum( ( D.t .- y.([model], D.x) ) .* model.ϕ.(D.x) )
    end

    q, hist = svgd_fit(q, grad_logp; alg_params...)
    return initial_dist, q, hist
end

function run_linear_regression(problem_params, alg_params, n_runs)
    svgd_results = []
    estimation_results = []

    true_model = RegressionModel(problem_params[:true_ϕ], problem_params[:true_w], 
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
        # H₀ = Distributions.entropy(initial_dist)
        # EV = ( numerical_expectation( initial_dist, 
        #                               w -> logistic_log_likelihood(D,w) )
        #        + expectation_V(initial_dist, initial_dist) 
        #        + 0.5 * log( det(2π * problem_params[:Σ_initial]) )
        #       )
        # est_logZ = estimate_logZ(H₀, EV,
        #                     alg_params[:step_size] * sum( get(hist,:dKL)[2] ) 
        #                          )

        # push!(svgd_results, hist)
        # push!(estimation_results, est_logZ)
    end

    # file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    # tagsave(datadir(DIRNAME, file_prefix * ".bson"),
    #         merge(alg_params, problem_params, 
    #             @dict n_runs estimation_results svgd_results),
    #         safe=true, storepatch = false)
end

### Experiments - linear regression on 3 basis functions

alg_params = Dict(
    :step_size => 0.01,
    :n_iter => 50, 
    :n_particles => 50,
    :kernel_width => "median_trick"
)

problem_params = Dict(
    :n_samples => 4,
    :sample_range => [-3, 3],
    :true_ϕ => x -> [x, x^2, x^4],
    :true_w => [20, -17, π],
    :true_β => 20,
    :ϕ => x -> [x, x^2, x^4],
    :w => [10, 817, π],
    :μ_prior => [0., 0, 0],
    :Σ_prior => [1 0 0.; 0 1 0; 0 0 1],
    # :Σ_initial => [9. 0.5 1; 0.5 8 2;1 2 1.],
)

n_runs = 1

run_linear_regression(problem_params, alg_params, n_runs)
