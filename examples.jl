# 1D Gaussian SVGD using Distributions package
#

# general params
n_particles = 50
e = 1
r = 1
n_iter = 100

μ₁ = -2
Σ₁ = 1
μ₂ = 2
Σ₂ = 1

# target mixture
d = MixtureModel(Normal[ Normal(μ₁, Σ₁), Normal(μ₂, Σ₂) ], [1/3, 2/3])
target = d
p = rand(target, n_particles)
tglp(x) = gradp(d, x)

#= dist(x) = 1/3. * normal_dist(x, μ₁, Σ₁) + 2/3. * normal_dist(x, μ₂, Σ₂) =#
#= tglp(x) = gradient(x -> log(dist(x)), x)[1] =#

# initial guess is a simple gaussian far from the target
initial_dist = Normal(-10)
q_0 = rand(initial_dist, (1, n_particles) )
q = copy(q_0)

#run svgd
@time q = svgd_fit(q, tglp, n_iter=200, repulsion=r, step_size=e)	

Plots.histogram(
                [reshape(q_0,n_particles) reshape(q,n_particles) reshape(p,n_particles)],	
    bins=15,
    alpha=0.5
   )


# 2d gaussian
n_particles = 50
e = 1
r = 1
n_iter = 100

# target
μ = [5, 3]
sig = [9. 0.5; 0.5 1.]  # MvNormal can deal with Int type means but not covariances
d = MvNormal(μ, sig)
target = d
p = rand(target, n_particles)
tglp(x) = gradp(d, x)

# initial
μ = [-2, -2]
sig = [1. 0.; 0. 1.]  # MvNormal can deal with Int type means but not covariances
initial = MvNormal(μ, sig)
q_0 = rand(initial, n_particles)
q = copy(q_0)

@time q = svgd_fit(q, tglp, n_iter=200, repulsion=r, step_size=e)	

Plots.scatter([q_0[1,:] q[1,:] p[1,:]],[q_0[2,:] q[2,:] p[2,:]], markercolor=["blue" "red" "green"], legend=:none)


# 2d gaussian mixture
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
tglp(x) = gradp(d, x)

# initial
μ = [-2, -2]
sig = [1. 0.; 0. 1.]  # MvNormal can deal with Int type means but not covariances
initial = MvNormal(μ, sig)
q_0 = rand(initial, n_particles)
q = copy(q_0)

@time q = svgd_fit(q, tglp, n_iter=400, repulsion=r, step_size=e)	

Plots.scatter([q_0[1,:] q[1,:] p[1,:]],[q_0[2,:] q[2,:] p[2,:]], markercolor=["blue" "red" "green"], legend=:none)
