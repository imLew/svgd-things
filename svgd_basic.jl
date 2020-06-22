using Statistics
using KernelFunctions
using LinearAlgebra
using Random
using Zygote
using Distances


function median_trick(x)
    if size(x)[end] == 1
        return 1
    end
    d = Distances.pairwise(Euclidean(), x)
    Statistics.median(d)^2/log(size(x)[end])
end

grad(f,x,y) = gradient(f,x,y)[1]

function svgd_fit_with_int(q, glp ;n_iter=100, repulsion=1, step_size=1)
    i = 0
    int_dKL = 0
    dKL_steps = []
    kernel = TransformedKernel( SqExponentialKernel(), ScaleTransform( 1/sqrt(median_trick(q))))
    while i < n_iter
        println(i)
        i += 1
        kernel.transform.s .= median_trick(q)
        q, dKL = svgd_step_with_int(q, kernel, glp, step_size, repulsion)
        int_dKL += step_size*dKL
        push!(dKL_steps, dKL)
    end
    return q, int_dKL, dKL_steps
end

function svgd_step_with_int(X, kernel::Kernel, grad_logp, ϵ, repulsion)
    n = size(X)[end]
    k_mat = kernelmatrix(kernel, X)
    grad_k = kernel_grad_matrix(kernel, X)
    glp_mat = grad_logp_matrix(grad_logp, X)
    if n == 1
        X += ϵ/n *  glp_mat * k_mat 
    else
        X += ϵ/n * ( glp_mat * k_mat 
                    + repulsion * hcat( sum(grad_k, dims=2)... ) 
                   )
    end
    dKL = n^2\sum( k_mat .* ( glp_mat' * glp_mat + 2*size(X)[1]/kernel.transform.s[1] * ones(n,n)
                             - 4/kernel.transform.s[1]^2 * Distances.pairwise(SqEuclidean(), X, dims=2)
                   )
             )
    s = 0
    for x in eachcol(X)
        for y in eachcol(X)
            s += gradient(x->kernel(x,y), x)[1]'*grad_logp(y)
            s += gradient(y->kernel(x,y), y)[1]'*grad_logp(x)
        end
    end
    dKL += s/n^2
    Plots.display( Plots.scatter(X[1,:],X[2,:]))
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
    kernel = TransformedKernel( SqExponentialKernel(), ScaleTransform( 1/sqrt(median_trick(q))))
    while i < n_iter
        i += 1
        kernel.transform.s .= median_trick(q)
        q = svgd_step(q, kernel, glp, step_size, repulsion)
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
    if size(X)[end] == 1
        return 0
    end
	mapslices(x -> grad.(kernel, [x], eachcol(X)), X, dims = 1)
end

