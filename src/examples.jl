using Plots
using Distributions
using LinearAlgebra
using PDMats

export plot_1D
export plot_2D
export run_svgd_and_plot
export expectation_V
export estimate_logZ
export logZ
export svgd_sample_from_known_dsitribution
export plot_known_dists

function svgd_sample_from_known_distribution(initial_dist, target_dist;
                                             alg_params)
    grad_logp(x) = gradp(target_dist, x)
    q = rand( initial_dist, alg_params[:n_particles] ) 
    if length(size(q)) == 1
        q = reshape(q, (1, length(q)))
    end
    q, hist = svgd_fit( q, grad_logp; alg_params... )
end

function expectation_V(initial_dist::Distribution, target_dist::Distribution) 
    numerical_expectation( initial_dist, x -> pdf_potential(target_dist, x) )
end

function expectation_V(initial_dist::Normal, target_dist::Normal)
    μ₀, σ₀ = params(initial_dist)
    μₚ, σₚ = params(target_dist)
    0.5 * ( σ₀^2 / σₚ^2 +  (μ₀-μₚ)^2/σₚ^2  )
end

function expectation_V(initial_dist::MvNormal, target_dist::MvNormal)
    μ₀, Σ₀ = params(initial_dist)
    μₚ, Σₚ = params(target_dist)
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

function logZ(d::T) where T <: Union{Normal, MvNormal}
    - logpdf( d, params(d)[1] )
end

function logZ(d::Exponential)
    1/λ
end

function pdf_potential(d::Distribution, x)
    -logpdf(d, x) # This potential is already normalized
end

function pdf_potential(d::Exponential, x)
    # Distribution.jl uses inverse param θ=1/λ (i.e. 1/θ e^{-x/θ})
    λ = 1/params(d)[1] 
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

function run_svgd_and_plot(initial_dist, target_dist, alg_params)
    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist)
    
    @time q_0, q, p, dKL = svgd_fit(initial_dist, target_dist; 
                                                   alg_params...)	

    @info "empirical logZ" estimate_logZ(H₀, EV, 
                                         alg_params[:step_size]*sum(dKL))
    @info "true logZ" logZ(target_dist)

    plot_results(initial_dist, target_dist, alg_params, H₀, 
                 logZ(target_dist), EV, dKL, q)
end

function plot_known_dists(initial_dist, target_dist, alg_params, 
                      H₀, logZ, EV, dKL, q)
    # caption="""n_particles=$n_particles; n_iter=$n_iter; 
    #         norm_method=$norm_method; kernel_width=$kernel_width; 
    #         step_size=$step_size"""
    caption = ""
    caption_plot = plot(grid=false,annotation=(0.5,0.5,caption),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);
    # title = """$(typeof(initial_dist)) $(Distributions.params(initial_dist)) 
    #          target $(typeof(target_dist)) 
    #          $(Distributions.params(target_dist))"""
    title = ""
    title_plot = plot(grid=false,annotation=(0.5,0.5,title),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);
    int_plot, norm_plot = plot_integration(H₀, logZ, EV, dKL, 
                                           alg_params[:step_size])

    dim = size(q)[1]
    if dim > 3 
        layout = @layout [t{0.1h} ; d{0.3w} i ; c{0.1h}]
        display(plot(title_plot, norm_plot, int_plot, 
                      caption_plot, layout=layout, size=(1400,800), 
                      legend=:topleft));
    else
        if dim == 1
            dist_plot = plot_1D(initial_dist, target_dist, q)
        elseif dim == 2
            dist_plot = plot_2D(initial_dist, target_dist, q)
        # elseif dim == 3
        #     dist_plot = plot_3D(initial_dist, target_dist, q)
        end
    layout = @layout [t{0.1h} ; i ; a b; c{0.1h}]
    plot(title_plot, int_plot, norm_plot, dist_plot, 
         caption_plot, layout=layout, size=(1400,800), 
         legend=:topleft);
    end
end

function plot_2D(initial_dist, target_dist, q)
    dist_plot = scatter(q[1,:], q[2,:], 
                        labels="q" , title="particles");
    min_q = minimum(q)
    max_q = maximum(q)
    t = min_q-0.5*abs(min_q):0.05:max_q+0.5*abs(max_q)
    contour!(t, t, (x,y)->pdf(target_dist, [x, y]), labels="p")
    contour!(t, t, (x,y)->pdf(initial_dist, [x, y]), labels="q₀")
    return dist_plot
end

function plot_integration(H₀, logZ, EV, dKL, step_size)
    int_plot = plot(estimate_logZ.([H₀], [EV],
                    step_size*cumsum(dKL)), title="log Z = $logZ",
                    labels="estimate for log Z");

    hline!([logZ], labels="analytical value log Z");

    norm_plot = plot(-step_size*dKL, labels="RKHS norm", title="dKL/dt");
    return int_plot, norm_plot
end

function plot_1D(initial_dist, target_dist, q)
    n_bins = length(q) ÷ 5
    dist_plot = histogram(reshape(q, length(q)), 
                          fillalpha=0.3, labels="q" ,bins=20,
                          title="particles", normalize=true);
    min_q = minimum(q)
    max_q = maximum(q)
    t = min_q-0.2*abs(min_q):0.05:max_q+0.2*abs(max_q)
    plot!(x->pdf(target_dist, x), t, labels="p")
    plot!(x->pdf(initial_dist, x), t, labels="q₀")
    return dist_plot
end

## Sampling
function gaussian_1d_mixture(;n_particles=100, step_size=1, n_iter=500,
                                μ₁ = -2, Σ₁ = 1, μ₂ = 2, Σ₂ = 1,
                               norm_method="standard")
    target = MixtureModel(Normal[ Normal(μ₁, Σ₁), Normal(μ₂, Σ₂) ], [1/3, 2/3])
    p = rand(target, n_particles)
    grad_logp(x) = gradp(target, x)
    initial_dist = Normal(-10)
    q_0 = rand(initial_dist, (1, n_particles) )
    q = copy(q_0)
    @time q, dkl = svgd_fit(q, grad_logp, n_iter=n_iter, 
                                           step_size=step_size, norm_method=norm_method)	
    q, q_0, p, dkl
end

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

    @time q, int_dkl, dkl = svgd_fit(q, tglp, n_iter=n_iter, step_size=e)	
    #= @time q = svgd_fit(q, tglp, n_iter=400, step_size=e) =#	
    plots.plot(dkl)
end

function gaussian3d_mixture()
    n_particles = 50
    e = 1
    r = 1
    n_iter = 100

    # target
    μ₁ = [5, 3, 0.]
    Σ₁ = [9. 0.5 1; 0.5 8 2;1 2 1.]  
    μ₂ = [7, 0, 1]
    Σ₂ = [1. 0 0; 0 1 0; 0 0 1.]  
    target = MixtureModel(MvNormal[ MvNormal(μ₁, Σ₁), 
                                   MvNormal(μ₂, Σ₂) ], [0.5, 0.5])
    p = rand(target, n_particles)
    grad_logp(x) = gradp(target, x)

    # initial
    μ = [-2, -2, -2]
    sig = [1. 0 0; 0 1 0; 0 0 1.]  
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
