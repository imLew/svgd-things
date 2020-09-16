using Plots
using Distributions

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
    :norm_method = "RKHS_norm",
    :kernel_width = "median_trick"
    )

