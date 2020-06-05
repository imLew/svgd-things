using Statistics
using KernelFunctions
using LinearAlgebra
using Random
using Zygote
using Distances


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

function svgd_step(X, kernel::Kernel, grad_logp, ϵ, repulsion)
    n = size(X)[end]
    k_mat = kernelmatrix(kernel, X)
    grad_k = kernel_grad_matrix(kernel, X)
    glp_mat = grad_logp_matrix(grad_logp, X)
    X += ϵ/n * ( glp_mat * k_mat 
                + repulsion * hcat( sum(grad_k, dims=2)... ) 
               )
end

function svgd_fit(q, glp ;n_iter=100, repulsion=1, step_size=1)
    i = 0
    while i < n_iter
        i += 1
        h = median_trick(q)
        k = TransformedKernel( SqExponentialKernel(), ScaleTransform( 1/sqrt(h)))
        q = svgd_step(q, k, glp, step_size, repulsion)
    end
    return q
end

function grad_logp_matrix(grad_logp, X)
    hcat( grad_logp.(eachcol(X))... )
end

function kernel_gradient(kernel::Kernel, x, y)
    gradient(x->kernel, x, y)
end

function kernel_grad_matrix(kernel::Kernel, X)
	mapslices(x -> grad.(kernel, [x], eachcol(X)), X, dims = 1)
end
