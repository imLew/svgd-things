#!/usr/bin/env julia
#$ -binding linear:16 # request cpus 
#$ -N gauss_to_gauss
#$ -q all.q 
#$ -cwd 
#$ -V 
#$ -t 1-96
using DrWatson
quickactivate(ENV["JULIA_ENVIRONMENT"], "SVGD")

global DIRNAME = "gaussian_to_gaussian"
global N_RUNS = 1

### local util functions
function gaussian_to_gaussian(;μ₀::Vector, μₚ::Vector, Σ₀::Matrix, Σₚ::Matrix,
                              alg_params)
    initial_dist = MvNormal(μ₀, Σ₀)
    target_dist = MvNormal(μₚ, Σₚ)
    q, hist = svgd_sample_from_known_distribution( initial_dist, target_dist; 
                                                 alg_params=alg_params )
    return initial_dist, target_dist, q, hist
end

# function gaussian_to_gaussian(;μ₀::Number, μₚ::Number, σ₀::Number, σₚ::Number,
#                               alg_params)
#     initial_dist = Normal(μ₀, σ₀)
#     target_dist = Normal(μₚ, σₚ)
#     q, hist = svgd_sample_from_known_distribution( initial_dist, target_dist; 
#                                                  alg_params=alg_params )
#     return initial_dist, target_dist, q, hist
# end

function run_g2g(;problem_params, alg_params, n_runs)
    svgd_results = []
    estimation_rkhs = []
    estimation_unbiased = []
    estimation_stein_discrep = []

    for i in 1:n_runs
        @info "Run $i/$(n_runs)"
        initial_dist, target_dist, q, hist = gaussian_to_gaussian( 
            ;problem_params..., alg_params=alg_params)

        H₀ = Distributions.entropy(initial_dist)
        EV = expectation_V( initial_dist, target_dist)

        est_logZ_rkhs = estimate_logZ(H₀, EV,
                        alg_params[:step_size] * sum( get(hist,:dKL_rkhs)[2] ) 
                                 )
        est_logZ_unbiased = estimate_logZ(H₀, EV,
                    alg_params[:step_size] * sum( get(hist,:dKL_unbiased)[2] ) 
                                 )
        est_logZ_stein_discrep = estimate_logZ(H₀, EV,
                alg_params[:step_size] * sum( get(hist,:dKL_stein_discrep)[2] ) 
                                 )
        global true_logZ = logZ(target_dist)

        push!(svgd_results, (hist, q))
        push!(estimation_rkhs, est_logZ_rkhs) 
        push!(estimation_unbiased, est_logZ_unbiased)
        push!(estimation_stein_discrep,est_logZ_stein_discrep)
               
    end

    file_prefix = savename( merge(problem_params, alg_params, @dict n_runs) )

    tagsave(datadir(DIRNAME, file_prefix * ".bson"),
            merge(alg_params, problem_params, 
                  @dict(n_runs, true_logZ, estimation_unbiased, 
                        estimation_stein_discrep,
                        estimation_rkhs, svgd_results)),
            safe=true, storepatch=true
    )
end

ALG_PARAMS = Dict(
    :n_iter => [2000, 5000],
    :step_size => [0.05, 0.01, 0.005],
    :n_particles => [ 50, 100, 200, 500],
    :norm_method => "RKHS_norm",
    :kernel_width => "median_trick"
)

PROBLEM_PARAMS_2D = Dict(
    :μ₀ => [[0., 0]],
    :μₚ => [[0,0.],[4,5]],
    :Σ₀ => [[1. 0; 0 1.]],
    :Σₚ => [[1. 0.5; 0.5 1],[2 0.1; 0.1 2]],
)

# TODO add params for a high dimensional gaussian
PROBLEM_PARAMS = PROBLEM_PARAMS_2D

if length(ARGS) == 0 && haskey(ENV, "SGE_TASK_ID")
# for running in a job array on the gridengine cluster;
# assumes dictionaries with parameters have been created in _research/tmp
# and that they are indexed in tmp_dict_names.bson
    using BSON
    using Distributions
    using SVGD
    dict_o_dicts = BSON.load(
                    projectdir("_research","tmp",
                        BSON.load(
                              projectdir("tmp_dict_names.bson")
                        )[ENV["SGE_TASK_ID"]][1]
                    )
                   )
    @info "Sampling problem: $(dict_o_dicts[:problem_params])"
    @info "Alg parameters: $(dict_o_dicts[:alg_params])"
    @time run_g2g(problem_params=dict_o_dicts[:problem_params],
                  alg_params=dict_o_dicts[:alg_params],
                  n_runs=dict_o_dicts[:N_RUNS])
elseif ARGS[1] == "make-dicts" 
# make dictionaries in a tmp directory containing the parameters for
# all the experiment we want to run
# also saves a dictionary mapping numbers 1 through #dicts to the dictionary
# names to index them
    dnames = Dict()
    for (i, alg_params) ∈ enumerate(dict_list(ALG_PARAMS))
        for (j, problem_params) ∈ enumerate(dict_list(PROBLEM_PARAMS))
            dname = tmpsave([@dict alg_params problem_params N_RUNS])
            dnames["$((i-1)*dict_list_count(PROBLEM_PARAMS) + j )"] = dname
        end
    end
    using BSON
    bson("tmp_dict_names.bson", dnames)
elseif ARGS[1] == "run"
# run the algorithm on the params specified in the second argument (bson)
    using BSON
    using Distributions
    using SVGD
    dict_o_dicts = BSON.load(ARGS[2])
    @info "Sampling problem: $(dict_o_dicts[:problem_params])"
    @info "Alg parameters: $(dict_o_dicts[:alg_params])"
    @time run_g2g(problem_params=dict_o_dicts[:problem_params],
                  alg_params=dict_o_dicts[:alg_params],
                  n_runs=dict_o_dicts[:N_RUNS])
elseif ARGS[1] == "run-all"
    Threads.@threads for file in readdir("_research/tmp", join=true)
        # run(`echo start`); run(`sleep 5`); run(`echo done`)
        run(`julia $PROGRAM_FILE run $file`)
    end
elseif ARGS[1] == "make-and-run-all"
# make the files containig the parameter dicts and start running them immediatly
    run(`julia $PROGRAM_FILE make-dicts`)
    run(`julia $PROGRAM_FILE run-all`)
end
