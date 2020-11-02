include("linear_regression.jl")
include("src/therm_int.jl")

# set up
n_dim = 3
problem_params = Dict(
    :n_samples => 20,
    :sample_range => [-3, 3],
    :true_ϕ => x -> [one(x), x, x^2],
    :true_w => [2, -1, 0.2],
    :true_β => 2,
    :ϕ => x -> [1, x, x^2],
    :μ_prior => zeros(n_dim),
    :Σ_prior => 0.05I(n_dim),
    :MAP_start => true,
)

true_model = RegressionModel(problem_params[:true_ϕ],
                             problem_params[:true_w], 
                             problem_params[:true_β])

D = generate_samples(model=true_model, n_samples=problem_params[:n_samples],
                     sample_range=problem_params[:sample_range])
scale = sum(extrema(D.t))
problem_params[:true_w] ./= scale
D.t ./= scale
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

function true_gauss_expectation(d::MvNormal, m::RegressionModel, D::RegressionData)
    X = reduce(hcat, m.ϕ.(D.x))
    @show size(X)
    0.5 * m.β * (tr((mean(d) * mean(d)' + cov(d)) * X * X')
        - 2 * D.t' * X' * mean(d)
        + D.t' * D.t
        + length(D) * log(m.β / 2π))
end

EV = ( true_gauss_expectation(initial_dist,  RegressionModel(problem_params[:ϕ], mean(initial_dist), 
problem_params[:true_β]), D)
        # num_expectation( 
            # initial_dist, 
            # w -> log_likelihood(D, RegressionModel(problem_params[:ϕ], w, 
                                                #    problem_params[:true_β])) )
       + SVGD.expectation_V(initial_dist, initial_dist) 
       + 0.5 * logdet(2π * problem_params[:Σ_prior])
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
                            problem_params[:true_ϕ], D.x)

@info "Value comparison" true_logZ 
@info "Value comparison" true_logZ therm_logZ est_logZ_rkhs est_logZ_stein_discrep est_logZ_unbiased

H₀ = Distributions.entropy(initial_dist)
EV = ( num_expectation( 
                    initial_dist, 
                    w -> log_likelihood(D, 
                            RegressionModel(problem_params[:ϕ], w, 
                                            problem_params[:true_β])) 
               )
               + SVGD.expectation_V(initial_dist, initial_dist) 
               + 0.5 * log( det(2π * problem_params[:Σ_prior]) )
              )
est_logZ = SVGD.estimate_logZ(H₀, EV,
                            alg_params[:step_size] * sum( get(hist,:dKL_rkhs)[2] ) 
                                 )

norm_plot = plot(hist[:phi_norm], title="φ norm", yaxis=:log);
int_plot = plot(
    SVGD.estimate_logZ.([H₀], [EV], alg_params[:step_size] * cumsum( get(hist, :dKL_rkhs)[2]))
    ,title="log Z", label="",
    );
fit_plot = plot_results(plot(), q, problem_params);
plot(fit_plot, norm_plot, int_plot, layout=@layout [f; n i])
