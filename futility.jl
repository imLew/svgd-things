n_particles=30
n_iter=400
norm_method="RKHS_norm"
step_size = 0.05
kernel_width = 0.5
q, q_0, p, rkhs_norm = run_and_plot(n_particles=n_particles, n_iter=n_iter, kernel_width=kernel_width=kernel_width, step_size=step_size)
# plot_svgd_results(q_0, q, p)

xi = q[:,i]
xj = q[:,j]
