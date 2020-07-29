module SVGD

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
    d = Distances.pairwise(Euclidean(), x, dims=2)
    Statistics.median(d)^2/log(size(x)[end])
end

grad(f,x,y) = gradient(f,x,y)[1]

function svgd_fit_with_int(q, glp ;n_iter=100, repulsion=1, step_size=1,
                          norm_method="standard")
    i = 0
    dKL_steps = []
    kernel = TransformedKernel( 
                SqExponentialKernel(), 
                ScaleTransform( 1/sqrt( median_trick(q) ) )
               )
    while i < n_iter
        @info "Step" i 
        i += 1
        #= kernel.transform.s .= 1/sqrt(median_trick(q)) =#
        q, dKL = svgd_step_with_int(q, kernel, glp, step_size, repulsion,
                                   norm_method=norm_method)
        push!(dKL_steps, dKL)
        # Plots.display(plot_svgd_results(q_0, q, p, title="$i"))
    end
    return q, dKL_steps
end

function svgd_step_with_int(X, kernel::KernelFunctions.Kernel, grad_logp, 
                            step_size, repulsion; norm_method="standard")
    n = size(X)[end]
    k_mat = kernelmatrix(kernel, X)
    grad_k = kernel_grad_matrix(kernel, X)
    glp_mat = hcat( grad_logp.(eachcol(X))... )
    if n == 1
        X += step_size/n *  glp_mat * k_mat 
    else
        X += step_size/n * ( glp_mat * k_mat 
                    + repulsion * hcat( sum(grad_k, dims=2)... ) 
                   )
    end
    @time dKL = compute_phi_norm(X, kernel, grad_logp, norm_method=norm_method)
    return X, dKL
end

function compute_phi_norm(X, kernel, grad_logp; norm_method="standard")
    if norm_method == "standard"
        stein_discrep_biased(X, kernel, grad_logp)
    elseif norm_method == "unbiased"
        unbiased_stein_discrep(X, kernel, grad_logp)
    # elseif method == "RKHS_norm"
    end
end
        

function unbiased_stein_discrep(X, kernel, grad_logp)
    n = size(X)[end]
    h = 1/kernel.transform.s[1]^2
    d = size(X)[1]
    dKL = 0
    for (i, x ) in enumerate(eachcol(X))
        for (j, y ) in enumerate(eachcol(X))
            if i != j
                dKL += kernel(x,y) * grad_logp(x)' * grad_logp(y)
                dKL += gradient(x->kernel(x,y), x)[1]'*grad_logp(y)
                dKL += gradient(y->kernel(x,y), y)[1]'*grad_logp(x)
                dKL += kernel(x,y) * ( 2d/h - 4/h^2 * SqEuclidean()(x,y))
            end
        end
    end
    dKL /= n*(n-1)
end

function stein_discrep_biased(X, kernel, grad_logp) 
    n = size(X)[end]
    h = 1/kernel.transform.s[1]^2
    d = size(X)[1]
    dKL = 0
    for x  in eachcol(X)
        for y  in eachcol(X)
            dKL += kernel(x,y) * grad_logp(x)' * grad_logp(y)
            dKL += gradient(x->kernel(x,y), x)[1]'*grad_logp(y)
            dKL += gradient(y->kernel(x,y), y)[1]'*grad_logp(x)
            dKL += kernel(x,y) * ( 2*d/h - 4/h^2 * SqEuclidean()(x,y))
        end
    end
    dKL /= n^2
end

# function stein_discrep_biased(X, kernel, grad_logp)
#     n = size(X)[end]
#     h = 1/kernel.transform.s[1]^2
#     k_mat = kernelmatrix(kernel, X)
#     grad_k = kernel_grad_matrix(kernel, X)
#     glp_mat = hcat( grad_logp.(eachcol(X))... )
#     dKL = n^2\sum( k_mat .* ( glp_mat' * glp_mat + 2*size(X)[1]/h * ones(n,n)
#                              - 4/h^2 * Distances.pairwise(SqEuclidean(), X, dims=2)
#                    )
#              )
#     s = 0
#     for x in eachcol(X)
#         for y in eachcol(X)
#             s += gradient(x->kernel(x,y), x)[1]'*grad_logp(y)
#             s += gradient(y->kernel(x,y), y)[1]'*grad_logp(x)
#         end
#     end
#     dKL += s/n^2
# end

# function svgd_step(X, kernel::KernelFunctions.Kernel, grad_logp, step_size, repulsion)
#     n = size(X)[end]
#     k_mat = kernelmatrix(kernel, X)
#     grad_k = kernel_grad_matrix(kernel, X)
#     glp_mat = ghcat( grad_logp.(eachcol(X))... )
#     X += step_size/n * ( glp_mat * k_mat 
#                 + repulsion * hcat( sum(grad_k, dims=2)... ) 
#                )
# end

# function svgd_fit(q, glp ;n_iter=100, repulsion=1, step_size=1)
#     i = 0
#     kernel = TransformedKernel( SqExponentialKernel(), ScaleTransform( 1/sqrt(median_trick(q))))
#     while i < n_iter
#         i += 1
#         kernel.transform.s .= median_trick(q)
#         q = svgd_step(q, kernel, glp, step_size, repulsion)
#     end
#     return q
# end

#= function grad_logp_matrix(grad_logp, X) =#
#=     hcat( grad_logp.(eachcol(X))... ) =#
#= end =#

#= function kernel_gradient(kernel::KernelFunctions.Kernel, x, y) =#
#=     gradient(x->kernel(x,y), x) =#
#= end =#


function kernel_grad_matrix(kernel::KernelFunctions.Kernel, X)
    if size(X)[end] == 1
        return 0
    end
	mapslices(x -> grad.(kernel, [x], eachcol(X)), X, dims = 1)
end

end # module

