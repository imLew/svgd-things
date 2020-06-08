using Statistics
using KernelFunctions
using LinearAlgebra
using Random
using Zygote
using Distances


function median_trick(x)
    d = Distances.pairwise(Euclidean(), x)
    Statistics.median(d)^2/log(size(x)[end])
end

grad(f,x,y) = gradient(f,x,y)[1]


function svgd_fit_with_int(q, glp ;n_iter=100, repulsion=1, step_size=1)
    i = 0
    int_dKL = 0
    k = TransformedKernel( SqExponentialKernel(), ScaleTransform( 1/sqrt(h)))
    while i < n_iter
        i += 1
        k.transform.s .= median_trick(q)
        q, dKL = svgd_step_with_int(q, k, glp, step_size, repulsion)
        int_dKL += dKL
    end
    return q, int_dKL
end

function svgd_step_with_int(X, kernel::Kernel, grad_logp, ϵ, repulsion)
    n = size(X)[end]
    k_mat = kernelmatrix(kernel, X)
    grad_k = kernel_grad_matrix(kernel, X)
    glp_mat = grad_logp_matrix(grad_logp, X)
    X += ϵ/n * ( glp_mat * k_mat 
                + repulsion * hcat( sum(grad_k, dims=2)... ) 
               )

    dKL = sum( k_mat * ( glp_mat' * glp_mat + 2*size(X)[1]/h * ones(n,n)
                    - 4/h^2 * Distances.pairwise(SqEuclidean(), X)
                   )
             )
    return X, dKL
end

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
    k = TransformedKernel( SqExponentialKernel(), ScaleTransform( 1/sqrt(h)))
    while i < n_iter
        i += 1
        k.transform.s .= median_trick(q)
        q = svgd_step(q, k, glp, step_size, repulsion)
    end
    return q
end

function grad_logp_matrix(grad_logp, X)
    hcat( grad_logp.(eachcol(X))... )
end

function kernel_gradient(kernel::Kernel, x, y)
    gradient(x->kernel(x,y), x)
end

function kernel_grad_matrix(kernel::Kernel, X)
	mapslices(x -> grad.(kernel, [x], eachcol(X)), X, dims = 1)
end

h = 1
k = TransformedKernel( SqExponentialKernel(), ScaleTransform( 1/sqrt(h)))
