using DrWatson
using BSON
using LinearAlgebra
using Distributions

using SVGD

global DIRNAME = "bayesian_logistic_regression"

### local util functions
function generate_2class_samples_from_gaussian(;n₀, n₁, μ₀=0, μ₁=1, Σ₀=1, Σ₁=1, 
                                               n_dim=1)
    if n_dim == 1
        generate_2class_samples(n₀, n₁, Normal(μ₁, Σ₁), Normal(μ₀, Σ₀))
    else
        generate_2class_samples(n₀, n₁, MvNormal(μ₁, Σ₁), MvNormal(μ₀, Σ₀))
    end
end

function generate_2class_samples(n₀, n₁, dist_0, dist_1)
    return [ones(Int(n₁)) rand(dist_1, n₁)'; zeros((n₀)) rand(dist_0, n₀)']
end

σ(a) = 1 / (1 + exp(-a))

function logistic_log_likelihood(D, w)
    t = D[:,1]
    x = D[:,2:end]
    z = [ones(size(x)[1]) x]
    sum( σ.(z*w) )
end

function logistic_grad_logp(D, w)
    y = D[:,1]
    x = D[:,2:end]
    z = [ones(size(x)[1]) x]
    sum((y .- σ.(z*w)).*z, dims=1)'
end

function fit_logistic_regression(problem_params, alg_params, D) 
    if problem_params[:n_dim] == 1
        initial_dist = Normal(problem_params[:μ_initial],
                              problem_params[:Σ_initial])
    else
        initial_dist = MvNormal(problem_params[:μ_initial],
                                problem_params[:Σ_initial])
    end
    q = rand(initial_dist, alg_params[:n_particles])
    grad_logp(w) = vec( - inv(problem_params[:Σ_initial])
                     * ( w-problem_params[:μ_initial] ) 
                     + logistic_grad_logp(D, w)
                    )

    q, hist = svgd_fit(q, grad_logp; alg_params...)
    return initial_dist, q, hist
end

function run_log_regression(;problem_params, alg_params, n_runs)
    svgd_results = []
    estimation_results = []

    # dataset with labels
    D = generate_2class_samples_from_gaussian(n₀=problem_params[:n₀],
                                              n₁=problem_params[:n₁],
                                              μ₀=problem_params[:μ₀],
                                              μ₁=problem_params[:μ₁], 
                                              Σ₀=problem_params[:Σ₀],
                                              Σ₁=problem_params[:Σ₁],
                                              n_dim=problem_params[:n_dim], 
                                             )

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        initial_dist, q, hist = fit_logistic_regression(problem_params, 
                                                        alg_params, D)
        H₀ = Distributions.entropy(initial_dist)
        EV = ( numerical_expectation( initial_dist, 
                                      w -> logistic_log_likelihood(D,w) )
               + expectation_V(initial_dist, initial_dist) 
               + 0.5 * log( det(2π * problem_params[:Σ_initial]) )
              )
        est_logZ = estimate_logZ(H₀, EV,
                            alg_params[:step_size] * sum( get(hist,:dKL)[2] ) 
                                 )

        push!(svgd_results, hist)
        push!(estimation_results, est_logZ)
    end

    file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                @dict n_runs estimation_results svgd_results),
            safe=true, storepatch = false)
end

### Experiments - logistic regression on 2D vectors
ALG_PARAMS = Dict(
    :step_size => [0.05, 0.01, 0.005 ],
    :n_iter => [ 50, 100 ],
    :n_particles => [ 50, 100],
    :norm_method => "RKHS_norm",
    :kernel_width => [ "median_trick", 0.5]
    )

PROBLEM_PARAMS = Dict(
    :n_dim => 2,
    :n₀ => 100,
    :n₁ => [ 50, 100 ],
    :μ₀ => [ [0., 0] ],
    :μ₁ => [ [1., 0.5], [5, 1] ],
    :Σ₀ => [ [1. 0.5; 0.5 1], [.5 0.1; 0.1 0.2] ],
    :Σ₁ => [ [1. 0.5; 0.5 1], [5 0.1; 0.1 2] ],
    :μ_initial => [ [0., 0, 0] ],
    :Σ_initial => [ [9. 0.5 1; 0.5 8 2;1 2 1.],  [1. 0 0; 0 1 0; 0 0 1.]  ],
    )

N_RUNS = 1

n_alg = dict_list_count(ALG_PARAMS)
n_prob = dict_list_count(PROBLEM_PARAMS)
@info "$(n_alg*n_prob) total experiments"
for (i, pp) ∈ enumerate(dict_list(PROBLEM_PARAMS)), 
        (j, ap) ∈ enumerate(dict_list(ALG_PARAMS))
    @info "Experiment $( (i-1)*n_alg + j) of $(n_alg*n_prob)"
    @info "Sampling problem: $pp"
    @info "Alg parameters: $ap"
    @time run_log_regression(
            problem_params=pp,
            alg_params=ap,
            n_runs=N_RUNS
            )
end
