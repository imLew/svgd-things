using DrWatson
using BSON
using ValueHistories
using Plots; plotly();
using Distributions

global DIRNAME = "gaussian_to_gaussian"

### local util functions
function gaussian_to_gaussian(;μ₀::Vector, μₚ::Vector, Σ₀::Matrix, Σₚ::Matrix,
                              alg_params)
    initial_dist = MvNormal(μ₀, Σ₀)
    target_dist = MvNormal(μₚ, Σₚ)
    q, hist = svgd_sample_from_known_dsitribution( initial_dist, target_dist; 
                                                 alg_params=alg_params )
    return initial_dist, target_dist, q, hist
end

function gaussian_to_gaussian(;μ₀::Number, μₚ::Number, σ₀::Number, σₚ::Number,
                              alg_params)
    initial_dist = Normal(μ₀, σ₀)
    target_dist = Normal(μₚ, σₚ)
    q, hist = svgd_sample_from_known_dsitribution( initial_dist, target_dist; 
                                                 alg_params=alg_params )
    return initial_dist, target_dist, q, hist
end

function run_g2g(;problem_params, alg_params, n_runs)
    svgd_results = []
    estimation_results = []

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        initial_dist, target_dist, q, hist = gaussian_to_gaussian( 
            ;problem_params..., alg_params=alg_params)

        H₀ = Distributions.entropy(initial_dist)
        EV = expectation_V( initial_dist, target_dist)

        est_logZ = estimate_logZ(H₀, EV,
                            alg_params[:step_size] * sum( get(hist,:dKL)[2] ) 
                                 )
        global true_logZ = logZ(target_dist)

        plot_known_dists(initial_dist, target_dist, alg_params, H₀, 
                         true_logZ, EV, get(hist,:dKL)[2], q)
        fn = savename( merge(problem_params, alg_params, Dict(:run_no => i) ))
        mkpath(plotsdir(DIRNAME))
        savefig( plotsdir(DIRNAME, fn * ".html") )
        push!(svgd_results, hist)
        push!(estimation_results, est_logZ)
    end

    file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                @dict n_runs true_logZ estimation_results svgd_results),
            safe=true, storepatch = false)
end

### Experiments
ALG_PARAMS = Dict(
    :step_size => [0.05, 0.01, 0.005 ],
    :n_iter => [ 50, 100 ],
    :n_particles => [ 50, 100],
    :norm_method => "RKHS_norm",
    :kernel_width => [ "median_trick", 0.5]
    )

PROBLEM_PARAMS_1D = Dict(
    :μ₀ => [-5, -2, 0],
    :μₚ => [0],
    :σ₀ => [0.5, 3],
    :σₚ => [0.5, 1, 2],
    )

PROBLEM_PARAMS_2D = Dict(
    :μ₀ => [-5, -2, 0],
    :μₚ => [0],
    :Σ₀ => [[1. 0; 0 1.],],
    :Σₚ => [[1. 0.5; 0.5 1],[0.2 0.1; 0.1 0.2]],
    )

PROBLEM_PARAMS_3D = Dict(
    :μ₀ => [[0, 0, 0.],[5, 3, 0.]],
    :μₚ => [0, 0, 0.],
    :Σ₀ => [[1. 0 0; 0 1 0; 0 0 1.]],
    :Σₚ => [[3. 0.5 1; 0.5 5 2;1 2 4.],[1. 0 0; 0 1 0; 0 0 1.]],
    )

n_alg = dict_list_count(ALG_PARAMS)
n_prob = dict_list_count(PROBLEM_PARAMS_1D)
@info "$(n_alg*n_prob) total experiments"
for (i, pp) ∈ enumerate(dict_list(PROBLEM_PARAMS_1D)), 
        (j, ap) ∈ enumerate(dict_list(ALG_PARAMS))
    @info "Experiment $( (i-1)*n_alg + j) of $(n_alg*n_prob)"
    @info "Sampling problem: $pp"
    @info "Alg parameters: $ap"
    @time run_g2g(
            problem_params=pp,
            alg_params=ap,
            n_runs=1
            )
    break
end
