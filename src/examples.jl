using Distributions
using LinearAlgebra
using PDMats

export gaussian_2d
export run_svgd_and_plot

export estimate_logZ
export expectation_V_gaussian

export numerical_expectation
export pdf_potential
export logZ

function expectation_V(q::Distribution, p::Distribution) 
    numerical_expectation( q, x -> pdf_potential(p, x) )
end

function expectation_V(q::T, p::T) where T <: Union{Normal, MvNormal}
    if Σ₀ isa Number
        return 0.5 * ( Σ₀ / Σₚ +  (μ₀-μₚ)^2/Σₚ  )
    end
    Σ₀ = get_pdmat(Σ₀)
    Σₚ = get_pdmat(Σₚ)
    0.5 * (  tr(inv(Σₚ)*Σ₀) + invquad(Σₚ, μ₀-μₚ) )
end

function estimate_logZ(H0, EV, int_KL)
    H0 - EV + int_KL
end

function numerical_expectation(d::Distribution, f; n_samples=10000)
    sum( f( rand(d, n_samples) ) ) / n_samples
end

function logZ(d::Distribution)
    println("log(Z) for distribution $d is not know, returning 0")
    return 0
end

function logZ(d::T where T <: Union{Normal, MvNormal})
    - logpdf( d, params(d)[1] )
end

function logZ(d::Exponential)
    1/λ
end

function pdf_potential(d::Distribution, x)
    -logpdf(d, x) # This potential is already normalized
end

function pdf_potential(d::Exponential, x)
    λ = 1/params(d)[1] # Distribution.jl uses inverse param θ=1/λ (i.e. 1/θ e^{-x/θ})
    λ * x
end

function pdf_potential(d::Normal, x)
    μ, σ = params(d)
    2 \ ((x-μ)/σ)^2
end

function pdf_potential(d::MvNormal, x)
    μ, Σ = params(d)
    2 \ invquad(Σ, x-μ)
end


function run_svgd(initial_dist, target_dist; alg_params)
    
    p = rand(target_dist, n_particles)
    grad_logp(x) = gradp(target_dist, x)
    q_0 = rand( initial_dist, (1, n_particles) )
    q = copy(q_0)
    @time q, rkhs_norm = svgd_fit_with_int(q, grad_logp, n_iter=n_iter, 
                                     step_size=step_size, norm_method=norm_method,
                                    kernel_width=kernel_width)	

    return q_0, q, p, rkhs_norm
end

function run_svgd_and_plot(initial_dist, target_dist; step_size=0.01, 
                           n_particles=100, n_iter=1000, 
                           norm_method="RKHS_norm", kernel_width="median_trick")

    H₀ = Distributions.entropy(initial_dist)
    EV = numerical_expectation( initial_dist, x -> -logpdf(target_dist, x) )
    logZ = 0
    
    @time q_0, q, p, rkhs_norm = run_svgd(initial_dist, target_dist, 
                                          alg_params=alg_params)	

    @info "empirical logZ" estimate_logZ(H₀, EV, step_size*sum(rkhs_norm))
    @info "true logZ" logZ

    layout = @layout [t{0.1h} ; i ; a b; c{0.1h}]

    caption="""n_particles=$n_particles; n_iter=$n_iter; norm_method=$norm_method; 
            kernel_width=$kernel_width; step_size=$step_size"""
    caption_plot = plot(grid=false,annotation=(0.5,0.5,caption),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);
    title = """$(typeof(initial_dist)) $(Distributions.params(initial_dist)) 
             target $(typeof(target_dist)) $(Distributions.params(target_dist))"""
    title_plot = plot(grid=false,annotation=(0.5,0.5,title),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);

    int_plot = plot(estimate_logZ.([H₀], [EV],
                        step_size*cumsum(rkhs_norm)), title="log Z = $logZ",
                    labels="estimate for log Z");

    hline!([logZ], labels="analytical value log Z");

    norm_plot = plot(-step_size*rkhs_norm, labels="RKHS norm", title="dKL/dt");

    distributions = histogram(reshape(q, length(q)), 
                              fillalpha=0.3, labels="q" ,bins=20,
                              title="particles", normalize=true);
    if target_dist != nothing
        plot!(x->pdf(target_dist, x),
              minimum(q)-0.2*abs(minimum(q)):0.05:maximum(q)+0.2*abs(maximum(q)),
              labels="p")
    end
    if initial_dist != nothing
        plot!(x->pdf(initial_dist, x),
              minimum(q)-0.2*abs(minimum(q)):0.05:maximum(q)+0.2*abs(maximum(q)),
              labels="q₀")
    end

    display( plot(title_plot, int_plot, norm_plot, distributions, caption_plot,
                  layout=layout, size=(1400,800), 
                  legend=:topleft
                 )
           );
end

## Sampling

function exponential_1D(;
                    n_particles=100,
                    step_size=1,
                    n_iter=500,
                    λ_initial = 5,
                    λ_target = 1,
                    norm_method="standard",
                    kernel_width=nothing
                   )
    target =  Exponential(1/λ_target)
    p = rand(target, n_particles)
    grad_logp(x) = gradp(target, x)
    initial_dist = Exponential(1/λ_initial)
    q_0 = rand(initial_dist, (1, n_particles) )
    q = copy(q_0)
    @time q, dkl = svgd_fit_with_int(q, grad_logp, n_iter=n_iter, 
                                     step_size=step_size, norm_method=norm_method,
                                    kernel_width=kernel_width)	
    q, q_0, p, dkl
end

function gaussian_1d(;
                    n_particles=100,
                    step_size=1,
                    n_iter=500,
                    μ_initial = 0,
                    Σ_initial = 0.5,
                    μ_target = 0,
                    Σ_target = 1,
                    norm_method="standard",
                    kernel_width=nothing
                   )
    target =  Normal(μ_target, sqrt( Σ_target ))
    p = rand(target, n_particles)
    grad_logp(x) = gradp(target, x)
    initial_dist = Normal(μ_initial, sqrt(Σ_initial))
    q_0 = rand(initial_dist, (1, n_particles) )
    q = copy(q_0)
    @time q, dkl = svgd_fit_with_int(q, grad_logp, n_iter=n_iter, 
                                     step_size=step_size, norm_method=norm_method,
                                    kernel_width=kernel_width)	
    q, q_0, p, dkl
end

function gaussian_1d_mixture(;n_particles=100, step_size=1, n_iter=500,
                                μ₁ = -2, Σ₁ = 1, μ₂ = 2, Σ₂ = 1,
                               norm_method="standard")
    target = MixtureModel(Normal[ Normal(μ₁, Σ₁), Normal(μ₂, Σ₂) ], [1/3, 2/3])
    p = rand(target, n_particles)
    grad_logp(x) = gradp(target, x)
    initial_dist = Normal(-10)
    q_0 = rand(initial_dist, (1, n_particles) )
    q = copy(q_0)
    @time q, dkl = svgd_fit_with_int(q, grad_logp, n_iter=n_iter, 
                                           step_size=step_size, norm_method=norm_method)	
    q, q_0, p, dkl
end

function gaussian_2d(;n_particles=100, step_size=0.05, n_iter=500,
                     μ_target=[-2, 8], Σ_target=[9. 0.5; 0.5 1.],
                     μ_initial = [0, 0], Σ_initial = [1. 0.; 0. 1.],
                    norm_method="standard")
    target = MvNormal(μ_target, Σ_target)
    p = rand(target, n_particles)
    grad_logp(x) = gradp(target, x)
    initial = MvNormal(μ_initial, Σ_initial)
    q_0 = rand(initial, n_particles)
    q = copy(q_0)
    q, dKL = svgd_fit_with_int(q, grad_logp, n_iter=n_iter, 
                               step_size=step_size,
                               norm_method=norm_method)
    return q, q_0, p, dKL
end

# TODO: fix this function
function gaussian2d_mixture()
    n_particles = 50
    e = 1
    r = 1
    n_iter = 500

    # target
    μ₁ = [5, 3]
    σ₁ = [9. 0.5; 0.5 1.]  # mvnormal can deal with int type means but not covariances
    μ₂ = [7, 0]
    σ₂ = [1. 0.1; 0.1 7.]  # mvnormal can deal with int type means but not covariances
    target = MixtureModel(MvNormal[ MvNormal(μ₁, σ₁), MvNormal(μ₂, σ₂) ], [0.5, 0.5])
    p = rand(target, n_particles)
    tglp(x) = gradp(target, x)

    # initial
    μ = [0, 0]
    sig = [1. 0.; 0. 1.]  # MvNormal can deal with int type means but not covariances
    initial = MvNormal(μ, sig)
    q_0 = rand(initial, n_particles)
    q = copy(q_0)

    @time q, int_dkl, dkl = svgd_fit_with_int(q, tglp, n_iter=n_iter, step_size=e)	
    #= @time q = svgd_fit(q, tglp, n_iter=400, step_size=e) =#	
    plots.plot(dkl)
end

# TODO: fix this function
function gaussian3d_mixture()
    n_particles = 50
    e = 1
    r = 1
    n_iter = 100

    # target
    μ₁ = [5, 3, 0.]
    Σ₁ = [9. 0.5 1; 0.5 8 2;1 2 1.]  # MvNormal can deal with Int type means but not covariances
    μ₂ = [7, 0, 1]
    Σ₂ = [1. 0 0; 0 1 0; 0 0 1.]  # MvNormal can deal with Int type means but not covariances
    target = MixtureModel(MvNormal[ MvNormal(μ₁, Σ₁), MvNormal(μ₂, Σ₂) ], [0.5, 0.5])
    p = rand(target, n_particles)
    grad_logp(x) = gradp(target, x)

    # initial
    μ = [-2, -2, -2]
    sig = [1. 0 0; 0 1 0; 0 0 1.]  # MvNormal can deal with Int type means but not covariances
    initial = MvNormal(μ, sig)
    q_0 = rand(initial, n_particles)
    q = copy(q_0)

    @time q = svgd_fit(q, grad_logp, n_iter=400, step_size=e)	

    plot_svgd_results(q_0, q, p)
end

## model fitting
# TODO: fix this function
function regression2d() 
    n_samples = 100
    dist_samples = 3
    ratio = 0.5
    n_iter = 100
    step_size = 0.1
    n_dim = 2
    data = generate_samples(n_samples, dist_samples, ratio, n_dim)

    q = randn(n_dim+1, n_samples)
    q0 = copy(q)
    grad_logp(w) = logistic_grad_logp(data, w)

    q = svgd_fit(q, grad_logp, n_iter=n_iter, step_size=step_size)

    traindata = copy(data)
    traindata[:, 1] = log_regr(data[:,2:end], mean(q, dims=2))

    plot_classes(traindata, data[:,1])
end
