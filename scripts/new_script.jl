using ForwardDiff
using Flux.Optimise
using KernelFunctions
using Plots
using Distributions
using Distances
using LinearAlgebra
using ProgressMeter


## Create the data
n_data = 100
n_dim = 1
x = rand(n_data, n_dim)
X = hcat(ones(n_data), x)
β = 0.5
prior = MvNormal(zeros(n_dim + 1), β^2 * I(n_dim + 1))
w_true = rand(prior)
σ = 0.1
y = X * w_true + randn(n_data) * σ
likelihood = MvNormal(y, σ^2)

## Utilitary functions

logp(w) = logpdf(likelihood, X * w) + logpdf(prior, w)

function V(w)
    diffval = mean(likelihood) - X * w
    0.5 * (dot(diffval, cov(likelihood) \ diffval) + dot(w, cov(prior) \ w))
end

gradlogp(w) = ForwardDiff.gradient(logp, w)

gradkernel(k, x, y) = ForwardDiff.gradient(x->k(x, y), x)

function plot_data(w)
    scatter(x, y, lab= "data")
    plot!(x, x->dot([1, x], w), lab ="y=x*w")
end

function plot_W(W)
    p = contour(range((mean(true_p)[1] .+ [-1, 1] * 5 * sqrt(var(true_p)[1]))..., length = 100),
    range((mean(true_p)[2] .+ [-1, 1] * 5 * sqrt(var(true_p)[2]))..., length=100),
    (x,y)->pdf(true_p, [x, y]),lw =3.0, colorbar = false)
    scatter!([w_true[1]], [w_true[2]], lab="Truth")
    scatter!(eachrow(W)..., lab="Particles")
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

function update_k!(k, X)
    D = pairwise(Euclidean(), X, dims = 2)
    D = LowerTriangular(D) - Diagonal(D)
    z = median(D[D.!=0])
    k.kernel.transform.s[1] = 1 / z
end

function optim_step(W, ϕ)
    εs = 10^(-9:0.1:0)
    tot_logp = zero(εs)
    for (i, ε) in enumerate(εs)
        tot_logp[i] = sum(V, eachcol(W .+ ε * ϕ))
    end
    return εs[argmin(tot_logp)]
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

function compute_dKL(k, X, ϕ)
    K = kernelmatrix(k, X, obsdim = 2) + 1e-8I
    dot(ϕ / K, ϕ)
end

true_p = true_posterior(X, y, β, σ)
true_Z = true_logevidence(X, y, β, σ)
## Training conditions
do_plot = false
freq = 1
n_iter = 1_000
n_particles = 20
n_samples = 100_000
k = 1.0 * transform(SqExponentialKernel(), 1.0)
m_init = mean(prior)
# m_init = mean(true_p)
C_init = cov(prior)
d_init = MvNormal(m_init, C_init)
W = rand(d_init, n_particles)
update_k!(k, W)
ϵ = 0.001
ϵₜ(t) = (100 + t)^(-0.5)
start = -20
slow_start(t) = 1.0#inv(1 + exp(-(t + start)))
thresh = 10.0* n_particles#Inf #0.1 * sqrt(n_particles)
# opt = ClipNorm(sqrt(n_particles)
opt = AdaMax(0.1)
opt = Descent(0.1)
ϕ = similar(W)
dKL = zeros(n_iter)
εs = zeros(n_iter)
expec_0 = sum(V, eachcol(W)) / n_particles
expec_0 = mean(V, eachcol(rand(d_init, n_samples)))
entrop = entropy(prior)
big_jump = zero(ϕ)
anim = Animation()
@showprogress for i in 1:n_iter
    update_k!(k, W)
    comp_ϕ!(ϕ, W, k, n_particles) # Compute ϕ
    ε = optim_step(W, ϕ) #ϵₜ(i)
    α = 1.0#renorm(ϕ, thresh)
    # Δϕ = Optimise.apply!(opt, W, slow_start(i) * ϕ) # Rescale ϕ
    Δϕ = ε * α * ϕ
    dKL[i] = compute_dKL(k, W, Δϕ) # Compute dKL/dt
    W .+= Δϕ # Update ϕ with additional rescaling
    if any(isnan.(W))
        error("Values went to NaN")
    end
    if dKL[i] > 10000
        big_jump .= Δϕ
        @info "Big jump happened!"
    end
    εs[i] = ε * α
    if do_plot
        if i % freq == 0
            p_data = plot_data(vec(mean(W, dims = 2)))
            p_W = plot_W(W)
            plot(p_data, p_W, title = "Iteration $i")
            frame(anim)
        end
    end
end
if do_plot gif(anim) end
emp_logZ = entrop - expec_0 + sum(dKL)
@info emp_logZ
p1 = bar(["Truth", "Stein Integration"], [true_Z, emp_logZ],lab="")
p2 = plot(1:length(dKL), entrop - expec_0 .+ cumsum(dKL), lab="Stein integration")
hline!([true_Z], lab = "truth")
p3 = plot(1:length(εs), εs, label = "ε")
plot(p1, p2, p3, layout = (1, 3), lw = 3.0) |> display

## Plot result

p_data = plot_data(vec(mean(W, dims = 2)))
p_W = plot_W(W)
plot(p_data, p_W)
