n_particles=1000
n_iter=1000
norm_method="RKHS_norm"
step_size = 0.01
kernel_width = 0.5

q, q_0, p, rkhs_norm = run_and_plot(n_particles=n_particles, n_iter=n_iter, kernel_width=kernel_width=kernel_width, step_size=step_size)
