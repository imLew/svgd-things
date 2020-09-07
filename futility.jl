using Plots
using Distributions

function run_gaussian(;
    μ₀ = 0,
    μₚ = 0,
    Σ₀ = 0.5^2,
    Σₚ = 1,
    n_particles=100,
    n_iter=1000,
    norm_method="RKHS_norm",
    step_size = 0.01,
    kernel_width = "median_trick",
    )

    q, q_0, p, rkhs_norm = gaussian_1d(
        n_particles=n_particles,
        step_size=step_size,
        n_iter=n_iter,
        μ_initial=μ₀,
        μ_target=μₚ,
        Σ_initial=Σ₀,
        Σ_target=Σₚ,
        norm_method=norm_method,
        kernel_width=kernel_width,
        )

    @info "final norm values" rkhs_norm[end-10:end]

    @info "empirical logZ" estimate_logZ(gaussian_entropy(μ₀, Σ₀), 
                                         expectation_V_gaussian(μ₀,μₚ, Σ₀, Σₚ),
                                         step_size*sum(rkhs_norm))
    @info "true logZ" gaussian_logZ(μₚ,Σₚ) 

    layout = @layout [t{0.1h} ; i ; a b; c{0.1h}]

    caption="""particles = $n_particles; iterations = $n_iter; 
            kernel_width = $kernel_width, norm_method = $norm_method; 
            step_size = $step_size; """ 
    caption_plot = plot(grid=false,annotation=(0.5,0.5,caption),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);
    title="Estimating log for 1D Gaussian μ₀ = $μ₀, μₚ = $μₚ, Σ₀ = $Σ₀, Σₚ = $Σₚ " 
    title_plot = plot(grid=false,annotation=(0.5,0.5,title),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);

    int_plot = plot( estimate_logZ.([gaussian_entropy(μ₀, Σ₀)], 
                                    [expectation_V_gaussian(μ₀,μₚ, Σ₀, Σₚ)],
                                    step_size*cumsum(rkhs_norm)), title="log Z",
                    labels="estimate for log Z");

    hline!([gaussian_logZ(μₚ,Σₚ)], labels="analytical value log Z");

    norm_plot = plot(-step_size*rkhs_norm, labels="RKHS norm", title="dKL/dt");

    distributions = histogram(reshape(q, length(q)), 
                              fillalpha=0.3, labels="q" ,bins=20,
                              title="particles", normalize=true);
    # plot!(

    # distributions = histogram([reshape(q_0, length(q_0)) reshape(q, length(q)) p],
    #                           fillalpha=0.3, labels=["q_0" "q" "p"],bins=20,
    #                           title="particles", normalize=true);

    display( plot(title_plot, int_plot, norm_plot, distributions, caption_plot,
                  layout=layout, size=(1400,800)));
end

run_gaussian(n_iter=1000, step_size=0.05)

run_gaussian(Σ₀=2, n_iter=2000)

run_gaussian(μ₀=-5, Σ₀=2, Σₚ=2, n_iter=2000, step_size=0.05)
