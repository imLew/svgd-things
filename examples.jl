using Plots
using PyPlot
using Distributions



## Sampling
begin # util functions
    function plot_svgd_results(q_0, q, p)
        d = size(q)[1]
        if d == 2
            Plots.scatter([q_0[1,:] q[1,:] p[1,:]],
                          [q_0[2,:] q[2,:] p[2,:]], 
                          markercolor=["blue" "red" "green"], 
                          label=["q_0" "q" "p"], 
                          legend = :outerleft)
        elseif d == 3
            Plots.scatter([q_0[1,:] q[1,:] p[1,:]],[q_0[2,:] q[2,:] p[2,:]],
                          [q_0[3,:] q[3,:] p[3,:]], 
                            markercolor=["blue" "red" "green"])
        end
    end

    function gradp(d::Distribution, x)
        if length(x) == 1
            x = x[1]
        end
        gradient(x->log(pdf(d, x)), reshape(x, length(x)) )[1]
    end

    function normal_dist(x, μ, Σ)
        if isa(x, Number)
            d = 1
        else
            d = size(μ)[1]
        end
        if d > 1
            1/ sqrt((2π)^d * LinearAlgebra.det(Σ)) * exp( -1/2 * (x - μ)' * (Σ \ (x - μ)) )
        else
            (sqrt(2π)*Σ) \ exp(-(x-μ)^2/Σ^2)
        end
    end
end


begin # 1D Gaussian SVGD using Distributions package
    # general params
    n_particles = 50
    e = 1
    r = 1
    n_iter = 100

    μ₁ = -2
    Σ₁ = 1
    μ₂ = 2
    Σ₂ = 1

    #= # target mixture =#
    target = MixtureModel(Normal[ Normal(μ₁, Σ₁), Normal(μ₂, Σ₂) ], [1/3, 2/3])
    p = rand(target, n_particles)
    tglp(x) = gradp(target, x)

    # initial guess is a simple gaussian far from the target
    initial_dist = Normal(-10)
    q_0 = rand(initial_dist, (1, n_particles) )
    q = copy(q_0)

    #run svgd
    @time q = svgd_fit(q, tglp, n_iter=100, repulsion=r, step_size=e)	

    Plots.histogram(
                    [reshape(q_0,n_particles) reshape(q,n_particles) reshape(p,n_particles)],	
        bins=15,
        alpha=0.5
       )
end

begin  # 2d gaussian
    n_particles = 50
    e = 1
    r = 1
    n_iter = 100

    # target
    μ = [-2, 8]
    sig = [9. 0.5; 0.5 1.]  # MvNormal can deal with Int type means but not covariances
    target = MvNormal(μ, sig)
    p = rand(target, n_particles)
    tglp(x) = gradp(target, x)

    # initial
    μ = [-2, -2]
    sig = [1. 0.; 0. 1.]  # MvNormal can deal with Int type means but not covariances
    initial = MvNormal(μ, sig)
    q_0 = rand(initial, n_particles)
    q = copy(q_0)

    @time q = svgd_fit(q, tglp, n_iter=100, repulsion=r, step_size=e)	

    Plots.scatter([q_0[1,:] q[1,:] p[1,:]],[q_0[2,:] q[2,:] p[2,:]], markercolor=["blue" "red" "green"], label=["q_0" "q" "p"], legend = :outerleft)
end

begin  # 2d gaussian mixture
    n_particles = 50
    e = 1
    r = 1
    n_iter = 100

    # target
    μ₁ = [5, 3]
    Σ₁ = [9. 0.5; 0.5 1.]  # MvNormal can deal with Int type means but not covariances
    μ₂ = [7, 0]
    Σ₂ = [1. 0.1; 0.1 7.]  # MvNormal can deal with Int type means but not covariances
    target = MixtureModel(MvNormal[ MvNormal(μ₁, Σ₁), MvNormal(μ₂, Σ₂) ], [0.5, 0.5])
    p = rand(target, n_particles)
    tglp(x) = gradp(target, x)

    # initial
    μ = [-1, -2]
    sig = [1. 0.; 0. 1.]  # MvNormal can deal with Int type means but not covariances
    initial = MvNormal(μ, sig)
    q_0 = rand(initial, n_particles)
    q = copy(q_0)

    @time q = svgd_fit(q, tglp, n_iter=400, repulsion=r, step_size=e)	

    plot_svgd_results(q_0, q, p)
end

begin  # 3d gaussian
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
    tglp(x) = gradp(target, x)

    # initial
    μ = [-2, -2, -2]
    sig = [1. 0 0; 0 1 0; 0 0 1.]  # MvNormal can deal with Int type means but not covariances
    initial = MvNormal(μ, sig)
    q_0 = rand(initial, n_particles)
    q = copy(q_0)

    @time q = svgd_fit(q, tglp, n_iter=400, repulsion=r, step_size=e)	

    plot_svgd_results(q_0, q, p)
end


# bayesian logistic regression

function generate_samples(n, dist, ratio, n_dim=1)
    n₁ = ceil(Int, ratio*n)
    n₂ = ceil(Int, (1-ratio)*n)
    x₁ = Random.randn(n₁,n_dim)
    x₂ = Random.randn(n₂,n_dim) .+ dist
    return [ones(n₁) x₁; zeros(n₂) x₂]
end

sigma(z,w) = 1 / (1 + exp(-1*sum(z'*w)))

function logistic_grad_logp(data, w)
    y = data[:,1]
    x = data[:,2:end]
    z = [ones(size(x)[1]) x]
    sum((y .- sigma.(eachrow(z),[w])).*z, dims=1)'
end


function logistic_regr_svgd(;n_iter=100, dim=2, num_particles=100, h=1, repulsion=0.5, e=0.1)
    n_samples = 100
    dist_samples = 3
    ratio = 0.5
    data = generate_samples(n_samples, dist_samples, ratio,dim)
    q = randn(dim+1, num_particles)
    q0 = copy(q)
    glp(w) = logistic_grad_logp(data, w)
    for i in 1:n_iter
        h = median_trick(q)
        k(x, y) = rbf(x, y, h)
        q = svgd_step(q, k, glp, e, repulsion)
    end
    return data, q0, q
end

function plot_classes(data)
    a_x = data[data[:,1].==0,2]
    a_y = data[data[:,1].==0,3]
    b_x = data[data[:,1].==1,2]
    b_y = data[data[:,1].==1,3]
    Plots.scatter([a_x, b_x], [a_y, b_y], markercolor=["blue" "red"], legend=:none)
end

function log_regr(X, w)
    Z = [ones(size(X)[1]) X]
    round.(Int, sigma.(eachrow(Z), [w]))
end

data, q0, q = logistic_regr_svgd(num_particles=1, n_iter=500)

traindata = data
traindata[:, 1] = log_regr(data[:,2:3], q)

plot_classes(traindata)
