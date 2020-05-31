using Statistics
using KernelFunctions
using LinearAlgebra
using Random
using Zygote
using Distances

function rbf(x, y, h)
    exp(-1/h * sum((x - y).^2))
end

function pairwise_dist(X)
    dist = []
    for (i, x) in enumerate(X)
        for (j, y) in enumerate(X)
            if j>i
                push!(dist, euclidean(x,y))
            end
        end
    end
    return dist
end

function median_trick(x)
    d = pairwise_dist(x)
    Statistics.median(d)^2/log(size(x)[end])
end

grad(f,x,y) = gradient(f,x,y)[1]

function gradp(d::Distribution, x)
    if length(x) == 1
        x = x[1]
    end
    gradient(x->log(pdf(d, x)), reshape(x, length(x)) )[1]
end

function svgd_step(X, kernel::Kernel, grad_logp, ϵ, repulsion)
    n = size(X)[end]
    k_mat = kernelmatrix(kernel, X)
    grad_k = kernel_grad_matrix(kernel, X)
    X += (
            ϵ/n * ( 
                hcat( grad_logp.(eachcol(X))... ) * k_mat 
                + repulsion * hcat( sum(grad_k, dims=2)... ) 
           )
         )
end

function svgd_fit(q, glp ;n_iter=100, repulsion=1, step_size=1)
    i = 0
    while i < n_iter
        i += 1
        h = median_trick(q)
        k(x, y) = rbf(x, y, h)
        q = svgd_step(q, k, glp, step_size, repulsion)
    end
    return q
end


function kernel_gradient(kernel::Kernel, x, y)
    gradient(x->kernel, x, y)
end

function kernel_grad_matrix(kernel::Kernel, X)
	mapslices(x -> grad.(kernel, [x], eachcol(X)), X, dims = 1)
end
