n_particles=100
n_iter=900
norm_method="RKHS_norm"
step_size = 0.05
kernel_width = 0.5
q, q_0, p, rkhs_norm = run_and_plot(n_particles=n_particles, n_iter=n_iter, kernel_width=kernel_width=kernel_width, step_size=step_size)
# plot_svgd_results(q_0, q, p)

s0 = 0.5
kld=sum(rkhs_norm)*step_size
H0 = 0.5*(log(2*pi*s0^2)+1);
mlogZest = -H0 + 0.5*s0^2 .- kld # estimate
mlogZex=-0.5*(log(2*pi)) # analytical result


xi = q[:,i]
xj = q[:,j]
