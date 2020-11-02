include("linear_regression.jl")
include("src/therm_int.jl")

# set up
n_dim = 4
problem_params = Dict(
    :n_samples => 20,
    :sample_range => [-3, 3],
    :true_ϕ => x -> [1, x, x^2],
    :true_w => [2, -1, 0.2],
    :true_β => 2,
    :ϕ => x -> [1, x, x^2],
    :μ_prior => zeros(n_dim),
    :Σ_prior => 1.0I(n_dim),
    :MAP_start => true,
)

true_model = RegressionModel(problem_params[:true_ϕ],
                             problem_params[:true_w], 
                             problem_params[:true_β])
D = generate_samples(model=true_model, n_samples=problem_params[:n_samples],
                     sample_range=problem_params[:sample_range])

###############################################################################
## ThermoIntegration
## alg params 
nSamples = 3000
nSteps = 30
tlZ = []
## alg
x = true_model.ϕ.(D.x)
prior = TuringDiagMvNormal(zeros(n_dim), ones(n_dim))
logprior(θ) = logpdf(prior, θ)
function loglikelihood(θ)
    (length(x)/2 * log(problem_params[:true_β]/2π) 
     - problem_params[:true_β]/2 * sum( (D.t .- dot.([θ], x)).^2 )
    )
end
θ_init = rand(n_dim)

alg = ThermoIntegration(nSamples = nSamples, nSteps=nSteps)
samplepower_posterior(x->loglikelihood(x) + logprior(x), n_dim, alg.nSamples)
therm_logZ = alg(logprior, loglikelihood, n_dim)
push!(tlZ, therm_logZ)

###############################################################################
## SVGD integration

alg_params = Dict(
    :step_size => 0.0001,
    :n_iter => 2000,
    :n_particles => 10,
    :kernel_width => "median_trick",
)
initial_dist, q, hist = fit_linear_regression(problem_params, alg_params, D)
H₀ = Distributions.entropy(initial_dist)
EV = ( num_expectation( 
            initial_dist, 
            w -> log_likelihood(D, RegressionModel(problem_params[:ϕ], w, 
                                                   problem_params[:true_β])) )
       + SVGD.expectation_V(initial_dist, initial_dist) 
       + 0.5 * log( det(2π * problem_params[:Σ_prior]) )
      )
est_logZ_rkhs = SVGD.estimate_logZ(H₀, EV,
                alg_params[:step_size] * sum( get(hist,:dKL_rkhs)[2] ) 
                         )
est_logZ_unbiased = SVGD.estimate_logZ(H₀, EV,
            alg_params[:step_size] * sum( get(hist,:dKL_unbiased)[2] ) 
                         )
est_logZ_stein_discrep = SVGD.estimate_logZ(H₀, EV,
        alg_params[:step_size] * sum( get(hist,:dKL_stein_discrep)[2] ) 
                         )

###############################################################################
## compare results

true_logZ = regression_logZ(problem_params[:Σ_prior], problem_params[:true_β], 
                            problem_params[:true_ϕ], D)

@info "Value comparison" true_logZ tlZ
@info "Value comparison" true_logZ therm_logZ est_logZ_rkhs est_logZ_stein_discrep est_logZ_unbiased
