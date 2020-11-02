using AdvancedHMC
using Trapz
using Distributions, DistributionsAD
using LinearAlgebra
using ForwardDiff

function samplepower_posterior(logπ, n_dim, nSamples) # From AdvancedHMC.jl README
    # Choose initial parameter value
    initial_θ = randn(n_dim)

    # Set the number of samples to warmup iterations
    n_adapts = 1_000

    # Define a Hamiltonian system
    metric = DiagEuclideanMetric(n_dim)
    hamiltonian = Hamiltonian(metric, logπ, ForwardDiff)

    # Define a leapfrog solver, with initial step size chosen heuristically
    initial_ϵ = find_good_stepsize(hamiltonian, initial_θ)
    integrator = Leapfrog(initial_ϵ)

    # Define an HMC sampler, with the following components
    #   - multinomial sampling scheme,
    #   - generalised No-U-Turn criteria, and
    #   - windowed adaption for step-size and diagonal mass matrix
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    samples, stats = sample(hamiltonian, proposal, initial_θ, nSamples, adaptor, n_adapts; progress=false)

    return samples
end

struct ThermoIntegration 
    nSamples::Int # Number of samples per integration steps
    nIntSteps::Int # Number of integration steps 
    schedule::Vector{Float64} # Integration schedule
end

function ThermoIntegration(; nSamples = 2000, nSteps = 30, schedule = ((1:nSteps)./nSteps).^5)
    ThermoIntegration(nSamples, nSteps, schedule)
end

function (alg::ThermoIntegration)(loglikelihood, logprior, n_dim; kwargs...)
    function temperedlogπ(t, loglikelihood, logprior) # Create the tempered log-joint
        return function logπ(x)
            t * loglikelihood(x) + logprior(x)
        end
    end
    logZs = zeros(Float64, alg.nIntSteps)
    for (i, t) in enumerate(alg.schedule) # Loop over all temperatures
        @info "Step $i/$(alg.nIntSteps)"
        θs = samplepower_posterior(temperedlogπ(t, loglikelihood, logprior), n_dim, alg.nSamples) # Get samples from power posterior via NUTS
        logZs[i] = sum(loglikelihood, θs) / alg.nSamples # Compute the expectation of the log-likelihood given the samples
    end
    return trapz(alg.schedule, logZs) # Compute the integral using trapezoids
end

# # model for polynomial regression
# ϕ(x) = [x, x^2, x^4, x^5]
# β = 2
# true_w = [2, -1, 0.2, 1]
# sample_range = [-3,3]

# n_dim = 4
# n_samples = 50
# x = ϕ.( rand(Uniform(sample_range...), n_samples) )
# target = dot.([true_w], x) .+ sqrt(β) \ randn(n_samples)
# prior = TuringDiagMvNormal(zeros(n_dim), ones(n_dim))
# logprior(θ) = logpdf(prior, θ)
# loglikelihood(θ) = length(x)/2 * log(β/2π) - β/2 * sum( (target .- dot.([θ], x)).^2 )
# # loglikelihood(θ) = sum(logpdf.(Normal.(y, 0.1), x * θ))
# θ_init = rand(n_dim)
# logprior(θ_init)
# loglikelihood(θ_init)

# alg = ThermoIntegration(nSamples = 3000)
# samplepower_posterior(x->loglikelihood(x) + logprior(x), n_dim, alg.nSamples)
# val = alg(logprior, loglikelihood, n_dim)
