module SVGD

export svgd_fit_with_int
export median_trick
export kernel_grad_matrix
export empirical_RKHS_norm
export compute_phi_norm
export unbiased_stein_discrep
export stein_discrep_biased
export svgd_step_with_int


using Statistics
using KernelFunctions
using LinearAlgebra
using Random
using Zygote
using Distances
using PDMats


function median_trick(x)
    if size(x)[end] == 1
        return 1
    end
    d = Distances.pairwise(Euclidean(), x, dims=2)
    Statistics.median(d)^2/log(size(x)[end])
end

grad(f,x,y) = gradient(f,x,y)[1]

function svgd_fit_with_int(q, grad_logp ;n_iter=100, step_size=1,
                          norm_method="standard", kernel_width=nothing)
    i = 0
    dKL_steps = []
    if kernel_width isa Number
        kernel = TransformedKernel( 
                    SqExponentialKernel(), 
                    ScaleTransform(1/sqrt(kernel_width))
                   )
    else
        kernel = TransformedKernel( 
                    SqExponentialKernel(), 
                    ScaleTransform( 1/sqrt( median_trick(q) ) )
                   )
    end
    while i < n_iter
        @info "Step" i 
        i += 1
        # kernel.transform.s .= 1/sqrt(median_trick(q))
        q, dKL = svgd_step_with_int(q, kernel, grad_logp, step_size, norm_method=norm_method)
        push!(dKL_steps, dKL)
        # Plots.display(plot_svgd_results(q_0, q, p, title="$i"))
    end
    return q, dKL_steps
end

function svgd_step_with_int(q, kernel::KernelFunctions.Kernel, grad_logp, 
                            step_size; norm_method="standard")
    @time ϕ = calculate_phi(kernel, q, grad_logp)
    @time dKL = compute_phi_norm(q, kernel, grad_logp, norm_method=norm_method, ϕ=ϕ)
    q .+= step_size*ϕ
    return q, dKL
end

function calculate_phi(kernel, q, grad_logp)
    ϕ = zero(q)
    for (i, xi) in enumerate( eachcol(q) )
        for xj in eachcol(q)
            d = kernel(xj, xi) * grad_logp(xj)
            ϕ[:, i] += d + gradient( x->kernel(xj, x), xi)
        end
    end
    ϕ /= n
end

function calculate_phi_vectorized(kernel, q, grad_logp)
    n = size(q)[end]
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    grad_k = kernel_grad_matrix(kernel, q)
    glp_mat = hcat( grad_logp.(eachcol(q))... )
    if n == 1  
        ϕ = glp_mat * k_mat 
    else
        ϕ =  1/n * ( glp_mat * k_mat 
                    + hcat( sum(grad_k, dims=2)... ) 
                   )
    end
end

function compute_phi_norm(q, kernel, grad_logp; norm_method="standard", ϕ=nothing)
    if norm_method == "standard"
        stein_discrep_biased(q, kernel, grad_logp)
    elseif norm_method == "unbiased"
        unbiased_stein_discrep(q, kernel, grad_logp)
    elseif norm_method == "RKHS_norm"
        if size(q)[1] == 1
            empirical_RKHS_norm(kernel, q, ϕ)
        end
    end
end

function empirical_RKHS_norm(kernel::Kernel, q, ϕ)
    k_mat = kernelpdmat(kernel, q)
    x = invquad(k_mat, reshape(ϕ, length(ϕ)))
    x[1]  # otherwise it's an array of array instead of array of floats
end

function unbiased_stein_discrep(q, kernel, grad_logp)
    n = size(q)[end]
    h = 1/kernel.transform.s[1]^2
    d = size(q)[1]
    dKL = 0
    for (i, x ) in enumerate(eachcol(q))
        for (j, y ) in enumerate(eachcol(q))
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

function stein_discrep_biased(q, kernel, grad_logp) 
    n = size(q)[end]
    h = 1/kernel.transform.s[1]^2
    d = size(q)[1]
    dKL = 0
    for x  in eachcol(q)
        for y  in eachcol(q)
            dKL += kernel(x,y) * grad_logp(x)' * grad_logp(y)
            dKL += gradient(x->kernel(x,y), x)[1]'*grad_logp(y)
            dKL += gradient(y->kernel(x,y), y)[1]'*grad_logp(x)
            dKL += kernel(x,y) * ( 2*d/h - 4/h^2 * SqEuclidean()(x,y))
        end
    end
    dKL /= n^2
end

function kernel_grad_matrix(kernel::KernelFunctions.Kernel, q)
    if size(q)[end] == 1
        return 0
    end
	mapslices(x -> grad.(kernel, [x], eachcol(q)), q, dims = 1)
end

end # module
