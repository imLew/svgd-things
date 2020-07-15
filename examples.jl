using Plots
using PyPlot
using Distributions
using DataFrames
using CSV



## Sampling
begin # util functions
    function plot_svgd_results(q_0, q, p; title)
        d = size(q)[1]
        if d == 2
            Plots.scatter([q_0[1,:] q[1,:] p[1,:]],
                          [q_0[2,:] q[2,:] p[2,:]], 
                          markercolor=["blue" "red" "green"], 
                          label=["q_0" "q" "p"], 
                          legend = :outerleft,
                          title=title)
        elseif d == 3
            Plots.scatter([q_0[1,:] q[1,:] p[1,:]],[q_0[2,:] q[2,:] p[2,:]],
                          [q_0[3,:] q[3,:] p[3,:]], 
                            markercolor=["blue" "red" "green"],
                            title=title)
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


function gaussian_1d_mixture(;n_particles=100, step_size=1, repulsion=1, n_iter=500,
                                μ₁ = -2, Σ₁ = 1, μ₂ = 2, Σ₂ = 1)
    target = MixtureModel(Normal[ Normal(μ₁, Σ₁), Normal(μ₂, Σ₂) ], [1/3, 2/3])
    p = rand(target, n_particles)
    grad_logp(x) = gradp(target, x)
    initial_dist = Normal(-10)
    q_0 = rand(initial_dist, (1, n_particles) )
    q = copy(q_0)
    q, dKL = svgd_fit_with_int(q, grad_logp, n_iter=n_iter, repulsion=repulsion, step_size=step_size)	
    return q, q_0, p, dKL
end

collect_stein_discrepancies(particle_sizes=[10, 50, 100, 200, 300], 
                            problem_function=gaussian_2d, 
                            dir_name="gaussian_1D_mixture", 
                            n_samples=10)



function plot_discrep(filename, func; n_iter=nothing, title=nothing, label=nothing)
    file = CSV.File(filename; header=false)
    df = DataFrame(file)
    n_tot = length(df[1])
    if n_iter == nothing
        n_iter = n_tot-1
    end
    Plots.plot([n_tot-n_iter : n_tot...], func(df)[end-n_iter:end], title=title,
    label=label)
end

Random.seed!(69)
Z = gaussian_2d(n_iter=500, step_size=0.5, norm_method="standard")

Random.seed!(69)
X = gaussian_2d(n_iter=500, step_size=0.5, norm_method="unbiased")

Plots.plot(X[4])

function gaussian_2d(;n_particles=100, step_size=1, repulsion=1, n_iter=500,
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
                               repulsion=repulsion, step_size=step_size,
                               norm_method=norm_method)
    return q, q_0, p, dKL
end

function collect_stein_discrepancies(;particle_sizes, problem_function, dir_name, 
                                     n_samples=100)
    result = Dict()
    particle_sizes = [10, 20, 50, 100]
    for n_particles in particle_sizes
        result["$n_particles"] = []
        i = 0
        while i<n_samples
            q, int_dKL, dKL = problem_function(n_particles=n_particles)
            push!(result["$n_particles"], dKL)
            @info "step" n_particles, i
            i += 1
        end
    end
    if isdir(dir_name)
        dir_name = dir_name*"_results"
    end
    mkdir(dir_name)
    for s in particle_sizes
        CSV.write(joinpath(dir_name,"$s"*"_particles"), 
                  DataFrame(result["$s"]), 
                  writeheader=false)
    end
end

result = Dict()
for n_particles in [10, 20, 50, 100]
    result["$n_particles"] = []
    i = 0
    @info "for n particles" n_particles
    while i < 100
        if n_particles == 100 && i < 90
            i = 90
        end
        begin  # 2d gaussian mixture
            # n_particles = 100
            e = 1
            r = 1
            n_iter = 500

            # target
            μ₁ = [5, 3]
            Σ₁ = [9. 0.5; 0.5 1.]  # MvNormal can deal with Int type means but not covariances
            μ₂ = [7, 0]
            Σ₂ = [1. 0.1; 0.1 7.]  # MvNormal can deal with Int type means but not covariances
            target = MixtureModel(MvNormal[ MvNormal(μ₁, Σ₁), MvNormal(μ₂, Σ₂) ], [0.5, 0.5])
            p = rand(target, n_particles)
            grad_logp(x) = gradp(target, x)

            # initial
            μ = [0, 0]
            sig = [1. 0.; 0. 1.]  # MvNormal can deal with Int type means but not covariances
            initial = MvNormal(μ, sig)
            q_0 = rand(initial, n_particles)
            q = copy(q_0)

            @time q, int_dKL, dKL = svgd_fit_with_int(q, grad_logp, n_iter=n_iter, repulsion=r, step_size=e)	
            #= @time q = svgd_fit(q, grad_logp, n_iter=400, repulsion=r, step_size=e) =#	
            Plots.plot(dKL)
            #= plot_svgd_results(q_0, q, p) =#
        end
        push!(result["$n_particles"], dKL)
        @info "step" i
        i += 1
    end
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
    grad_logp(x) = gradp(target, x)

    # initial
    μ = [-2, -2, -2]
    sig = [1. 0 0; 0 1 0; 0 0 1.]  # MvNormal can deal with Int type means but not covariances
    initial = MvNormal(μ, sig)
    q_0 = rand(initial, n_particles)
    q = copy(q_0)

    @time q = svgd_fit(q, grad_logp, n_iter=400, repulsion=r, step_size=e)	

    plot_svgd_results(q_0, q, p)
end


# bayesian logistic regression
begin  # regression util functions
    function generate_samples(n, dist, ratio, n_dim=2)
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

    function plot_classes(data, truth)
        # only 2D 
        x = traindata[:,2]
        y = traindata[:,3]
        l = traindata[:,1]
        t = data[:,1]
        t.!=y
        Plots.scatter([x[l.==t], x[l.!=t]], [y[l.==t], y[l.!=t]], label=["correct" "incorrect"])
        #TODO: color by average of prediction
    end

    function log_regr(X, w)
        Z = [ones(size(X)[1]) X]
        round.(Int, sigma.(eachrow(Z), [w]))
    end
end

begin  # 2D regression example
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
