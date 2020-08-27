using Statistics
using KernelFunctions
using LinearAlgebra
using Random
using Zygote
using Distances

using Plots
using Distributions
using DataFrames
using CSV
using ForwardDiff

export run_and_plot
export plot_svgd_results

function plot_svgd_results(q_0, q, p; title="", label="")
    d = size(q)[1]
    if d == 1
        histogram([q_0[:], q[:], p[:]], alpha=0.4,
                      markercolor=["blue" "red" "green"], 
                      label=["q_0" "q" "p"], 
                      legend = :outerleft,
                      title=title)
    elseif d == 2
        plot = Plots.scatter([q_0[1,:] q[1,:] p[1,:]],
                      [q_0[2,:] q[2,:] p[2,:]], 
                      markercolor=["blue" "red" "green"], 
                      label=["q_0" "q" "p"], 
                      legend = :outerleft,
                      title=title)
    elseif d == 3 
        plot = Plots.scatter([q_0[1,:] q[1,:] p[1,:]],[q_0[2,:] q[2,:] p[2,:]],
                      [q_0[3,:] q[3,:] p[3,:]], 
                        markercolor=["blue" "red" "green"],
                        title=title,
                        label=label)
    end
    # display(plot)
end

function gradp(d::Distribution, x)
    if length(x) == 1
        return Zygote.gradient(x->log(pdf.(d, x)[1]), x )[1]
    end
    ForwardDiff.gradient(x->log(pdf(d, x)), reshape(x, length(x)) )
end

function collect_stein_discrepancies(;particle_sizes, problem_function, dir_name, 
                                     n_samples=100)
    result = Dict()
    for n_particles in particle_sizes
        result["$n_particles"] = []
        i = 0
        while i<n_samples
            q, q_0, p, dKL = problem_function(n_particles=n_particles)
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
                  header=false,
                  append=true)
    end
end

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

function normal_dist(x, μ, Σ)
    if isa(x, Number)
        d = 1
    else
        d = size(μ)[1]
    end
    if d > 1
        1/ sqrt((2π)^d * LinearAlgebra.det(Σ)) * exp( -1/2 * (x - μ) * (Σ \ (x - μ)))
    else
        (sqrt(2π)*Σ) \ exp(-(x-μ)^2/Σ^2)
    end
end

function run_and_plot(; n_particles, n_iter, kernel_width, step_size=0.05,
                      problem=gaussian_1d, norm_method="RKHS_norm")

    q, q_0, p, rkhs_norm = problem(n_particles=n_particles, n_iter=n_iter, 
                norm_method=norm_method, kernel_width=kernel_width, step_size=step_size)
    int_value = sum(rkhs_norm)*step_size
    layout = @layout [t{0.1h} ; a b]

    @info "integral" int_value

    title="$n_particles particles; $n_iter iterations; kernel_width $kernel_width; integral dKL = $int_value"

    title_plot = plot(grid=false,annotation=(0.5,0.5,title),ticks=([]),fgborder=:white,subplot=1, framestyle=:none)

    norm_plot = plot(rkhs_norm, labels="RKHS norm");

    distributions = histogram([reshape(q_0, length(q_0)) reshape(q, length(q)) p], fillalpha=0.3, labels=["q_0" "q" "p"])

    display( plot(title_plot, norm_plot, distributions, layout=layout, size=(1400,800)))
    if occursin(".", title)
        title = join(split(title, "."), "point")
    end
    savefig("plots/$(join(split(title)))")
q, q_0, p, rkhs_norm
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