using Plots
using Distributions

function run_integration(;entropy, expectation_V, logZ, params, problem=gaussian_1d,
                        initial_dist=nothing, target_dist=nothing)

    step_size = params[:step_size]
    n_particles = params[:n_particles]
    n_iter = params[:n_iter]
    norm_method = params[:norm_method]
    kernel_width = params[:kernel_width]
    μ_initial = params[:μ_initial]
    Σ_initial = params[:Σ_initial]
    μ_target = params[:μ_target]
    Σ_target = params[:Σ_target]
    
    q, q_0, p, rkhs_norm = problem(;params...)

    @info "final norm values" rkhs_norm[end-10:end]

    @info "empirical logZ" estimate_logZ(entropy,
                                         expectation_V,
                                         step_size*sum(rkhs_norm))
    @info "true logZ" logZ

    layout = @layout [t{0.1h} ; i ; a b; c{0.1h}]

    caption="n_particles=$n_particles; n_iter=$n_iter; norm_method=$norm_method; kernel_width=$kernel_width; step_size=$step_size"
    # caption="$(params...)"
    caption_plot = plot(grid=false,annotation=(0.5,0.5,caption),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);
    title = ""
    title="""Estimating log for 1D Gaussian μ₀ = $μ_initial, μₚ = $μ_target, Σ₀ = $Σ_initial, Σₚ = $Σ_target """ 
    title_plot = plot(grid=false,annotation=(0.5,0.5,title),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);

    int_plot = plot( estimate_logZ.([entropy], 
                                    [expectation_V],
                                    step_size*cumsum(rkhs_norm)), title="log Z = $logZ",
                    labels="estimate for log Z");

    hline!([logZ], labels="analytical value log Z");

    norm_plot = plot(-step_size*rkhs_norm, labels="RKHS norm", title="dKL/dt");

    distributions = histogram(reshape(q, length(q)), 
                              fillalpha=0.3, labels="q" ,bins=20,
                              title="particles", normalize=true);
    if target_dist != nothing
        plot!(target_dist,
              minimum(q)-0.2*abs(minimum(q)):0.05:maximum(q)+0.2*abs(maximum(q)),
              labels="p")
    end
    if initial_dist != nothing
        plot!(initial_dist,
              minimum(q)-0.2*abs(minimum(q)):0.05:maximum(q)+0.2*abs(maximum(q)),
              labels="q₀")
    end

    display( plot(title_plot, int_plot, norm_plot, distributions, caption_plot,
                  layout=layout, size=(1400,800), 
                  legend=:topleft
                 )
           );
end


"""
fit a gaussian target with a gaussian initial distibution and
compute the estimate for log Z"""
function run_gaussian(;
    n_particles=50,
    step_size=0.01,
    n_iter=1000,
    μ_initial=0,
    μ_target=0,
    Σ_initial=1,
    Σ_target=1,
    norm_method="RKHS_norm",
    kernel_width="median_trick",
    )
    params = Dict(
    :n_particles=>n_particles,
    :step_size=>step_size,
    :n_iter=>n_iter,
    :μ_initial=>μ_initial,
    :μ_target=>μ_target,
    :Σ_initial=>Σ_initial,
    :Σ_target=>Σ_target,
    :norm_method=>norm_method,
    :kernel_width=>kernel_width,
    )

    run_integration(
        entropy=gaussian_entropy( μ_initial, Σ_initial),
        expectation_V=expectation_V_gaussian( μ_initial, Σ_initial, μ_target, Σ_target),
        params=params,
        logZ=gaussian_logZ(μ_target, Σ_target),
        problem=gaussian_1d,
        initial_dist=x -> pdf(Normal(μ_initial, Σ_initial), x),
        target_dist=x -> pdf(Normal(μ_target, Σ_target), x)
    )
end

"""
fit a exponential target with an exponential initial distibution and
compute the estimate for log Z"""
function run_exponential(;
    n_particles=50,
    step_size=0.01,
    n_iter=1000,
    λ_initial=0.2,
    λ_target=1,
    norm_method="RKHS_norm",
    kernel_width="median_trick",
    )
    params = Dict(
    :n_particles=>n_particles,
    :step_size=>step_size,
    :n_iter=>n_iter,
    :λ_initial=>λ_initial,
    :λ_target=>λ_target,
    :norm_method=>norm_method,
    :kernel_width=>kernel_width,
    )
    run_integration(
        entropy=exponential_entropy(λ_initial),
        expectation_V=numerical_expectation(
                            Exponential(1/λ_initial),
                            x->exponential_V(x, λ=λ_target)),
        params=params,
        logZ=exponential_logZ(λ_target),
        problem=exponential_1D,
        initial_dist=x -> pdf(Exponential(1/λ_initial), x),
        target_dist=x -> pdf(Exponential(1/λ_target), x)
    )
end

flatten(x) = reshape(x, length(x))

q, q0, p, rkhs_norm = gaussian_2d()

plot(flatten(rkhs_norm))

scatter([q0[1,:] q[1,:] p[1,:]],
          [q0[2,:] q[2,:] p[2,:]], 
          markercolor=["blue" "red" "green"], 
          label=["q_0" "q" "p"], 
          legend = :outerleft,)
          
problem_params = Dict(
    :initial_dist = :NdimGaussian,
    :initial_dist_params = (μ₀, Σ₀),
    :target_dist = :NdimGaussian,
    :target_dist_params = (μₚ, Σₚ),
    )

alg_params = Dict(
    :step_size = 0.01,
    :n_iter = 1000,
    :n_particles = 50,
    :kernel = :RBF,
    :kernel_params = h,
    :norm_method = "RKHS_norm"
    )
