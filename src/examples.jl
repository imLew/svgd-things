using Distributions

## Sampling

function gaussian_1d(;
                    n_particles=100,
                    step_size=1,
                    n_iter=500,
                    μ_initial = 0,
                    Σ_initial = 1,
                    μ_target = -9,
                    Σ_target = 0.8,
                    norm_method="standard",
                    kernel_width=nothing
                   )
    target =  Normal(μ_initial, Σ_initial)
    p = rand(target, n_particles)
    grad_logp(x) = gradp(target, x)
    initial_dist = Normal(μ_target, Σ_target)
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

function gaussian_2d(;n_particles=100, step_size=1, n_iter=500,
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
