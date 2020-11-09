using ForwardDiff
using Flux.Optimise
using KernelFunctions
using Plots
using Distributions, DistributionsAD
using LinearAlgebra
using ProgressMeter


## Create the data
n_data = 10
n_dim = 1
x = rand(n_data, n_dim)
X = hcat(ones(n_data), x)
β = 1.0
prior = MvNormal(zeros(n_dim + 1), β^2 * I(n_dim + 1))
w_true = rand(prior)
σ = 0.1
y = X * w_true + randn(n_data) * σ
logp(w) = logpdf(MvNormal(y, σ^2), X * w) + logpdf(prior, w)
gradlogp(w) = ForwardDiff.gradient(logp, w)
gradkernel(k, x, y) = ForwardDiff.gradient(x->k(x, y), x)
function plot_data(w)
    scatter(x, y, lab= "data")
    plot!(x, x->dot([1, x], w), lab ="y=x*w")
end
function plot_W(W)
    scatter(eachrow(W)..., lab="Particles")
    scatter!([w_true[1]], [w_true[2]], lab="Truth")
end
plot_data(w_true)

## True values
function true_posterior(X, y, β, σ)
    Σ = σ^2 * inv(X' * X + σ^2 / β^2 * I)
    μ = σ^2 \ Σ * X' * y
    MvNormal(μ, Σ)
end

function true_logevidence(X, y, β, σ)
    D = length(y)
    Ω = σ^2 * I + β^2 * X * X'
    -0.5 * (D * log(2π) + logdet(Ω) + dot(y, Ω \ y))
end

function comp_ϕ!(ϕ, X, k, n_particles)
    G = mapslices(gradlogp, X, dims = 1)
    for i in 1:n_particles
        ϕ[:, i] = n_particles \ sum(1:n_particles) do j
            G[:, j] * k(X[:, j], X[:, i]) + gradkernel(k, X[:, j], X[:, i])
        end
    end
end

function renorm(ϕ, thresh)
    if norm(ϕ) > thresh
        return thresh / norm(ϕ)
    else
        return 1.0
    end
end

function compute_dKL(k, X, ϕ, ϵ)
    K = kernelmatrix(k, X, obsdim = 2)
    - ϵ * dot(ϕ / K, ϕ)
end

true_p = true_posterior(X, y, β, σ)
true_Z = true_logevidence(X, y, β, σ)
## Training conditions
n_iter = 100
n_particles = 20
n_samples = 100_000
k = 1.0 * transform(SqExponentialKernel(), 1.0)
m_init = mean(prior)
m_init = mean(true_p)
C_init = cov(prior)
d_init = MvNormal(m_init, C_init)
W = rand(d_init, n_particles)
ϵ = 0.001
ϵₜ(t) = (100 + t)^(-1.0)
thresh = 2.0 * n_particles
opt = ClipNorm(1.0 * n_particles)
# opt = Momentum(1.0)
ϕ = similar(W)
dKL = zeros(n_iter)
expec_0 = sum(logp, eachcol(W)) / n_particles
expec_0 = mean(logp, eachcol(rand(d_init, n_samples)))
entrop = entropy(prior)

anim = Animation()
@showprogress for i in 1:n_iter
    comp_ϕ!(ϕ, W, k, n_particles) # Compute ϕ
    α = renorm(ϕ, thresh)
    ϵ = ϵₜ(i)
    # mϕ = Optimise.apply!(opt, W, ϕ) # Rescale ϕ
    W .+= ϵ * α * ϕ # Update ϕ with additional rescaling
    dKL[i] = compute_dKL(k, W, ϕ, ϵ * α) # Compute dKL/dt
    p_data = plot_data(vec(mean(W, dims = 2)))
    p_W = plot_W(W)
    plot(p_data, p_W)
    frame(anim)
end
gif(anim)
emp_logZ = entrop - expec_0 - sum(dKL)
@info emp_logZ
bar(["Truth", "Stein Integration"], [true_Z, emp_logZ])