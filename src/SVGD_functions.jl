using ProgressMeter
using Statistics
using ValueHistories
using KernelFunctions
using LinearAlgebra
using Random
using Zygote
using Distances
using PDMats

export svgd_fit
export median_trick
export kernel_grad_matrix
export empirical_RKHS_norm
export compute_phi_norm
export unbiased_stein_discrep
export stein_discrep_biased


function median_trick(x)
    if size(x)[end] == 1
        return 1
    end
    d = Distances.pairwise(Euclidean(), x, dims=2)
    median(d)^2/log(size(x)[end])
end

grad(f,x,y) = gradient(f,x,y)[1]

function svgd_fit(q, grad_logp ;n_iter=100, step_size=1,
                           norm_method="standard", kernel_width=nothing,
                           n_particles=50)
    hist = MVHistory()
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
    @showprogress for i in 1:n_iter
        if kernel_width == "median_trick"
            kernel.transform.s .= 1/sqrt(median_trick(q))
        elseif kernel_width == "median"
            kernel.transform.s .= 1/median(pairwise(Euclidean(), q, dims=2))
        elseif kernel_width == "median_squared"
            kernel.transform.s .= 1 / (0.002*median(pairwise(Euclidean(), q, dims=2))^2 )        
        elseif kernel_width == "mean"
            kernel.transform.s .= 1 / mean(pairwise(Euclidean(), q, dims=2))^2 
        end
        # ϕ = calculate_phi_vectorized(kernel, q, grad_logp)
        ϕ = calculate_phi(kernel, q, grad_logp)
        dKL_rkhs = compute_phi_norm(q, kernel, grad_logp, 
                                     norm_method="RKHS_norm", ϕ=ϕ)
        dKL_unbiased = compute_phi_norm(q, kernel, grad_logp, 
                                     norm_method="unbiased", ϕ=ϕ)
        dKL_stein_discrep = compute_phi_norm(q, kernel, grad_logp, 
                                     norm_method="standard", ϕ=ϕ)
        q .+= step_size*ϕ
        push!(hist, :dKL_unbiased, i, dKL_unbiased)
        push!(hist, :dKL_stein_discrep, i, dKL_stein_discrep)
        push!(hist, :dKL_rkhs, i, dKL_rkhs)
        push!(hist, :ϕ_norm, i, mean(norm(ϕ)))
    end
    return q, hist
end

function calculate_phi(kernel, q, grad_logp)
    glp = grad_logp.(eachcol(q))
    ϕ = zero(q)
    for (i, xi) in enumerate(eachcol(q))
        for (xj, glp_j) in zip(eachcol(q), glp)
            d = kernel(xj, xi) * glp_j
            # K = kernel_gradient( kernel, xj, xi )
            K = gradient( x->kernel(x, xi), xj )[1]
            ϕ[:, i] .+= d .+ K 
        end
    end
    ϕ ./= size(q)[end]
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

function compute_phi_norm(q, kernel, grad_logp; 
                          norm_method="standard", ϕ=nothing)
    if norm_method == "standard"
        stein_discrep_biased(q, kernel, grad_logp)
    elseif norm_method == "unbiased"
        unbiased_stein_discrep(q, kernel, grad_logp)
    elseif norm_method == "RKHS_norm"
        empirical_RKHS_norm(kernel, q, ϕ)
    end
end

function empirical_RKHS_norm(kernel::Kernel, q, ϕ)
    if size(q)[1] == 1
        invquad(kernelpdmat(kernel, q), vec(ϕ))
    else
        # this first method tries to flatten the tensor equation
        # invquad(flat_matrix_kernel_matrix(kernel, q), vec(ϕ))
        # the second method should be the straight forward case for a
        # kernel that is a scalar f(x) times identity matrix
        norm = 0
        k_mat = kernelpdmat(kernel, q)
        for f in eachrow(ϕ)
            norm += invquad(k_mat, vec(f))
        end
        return norm
    end
end

# function empirical_RKHS_norm(kernel::MatrixKernel, q, ϕ)
#     invquad(flat_matrix_kernel_matrix(kernel, q), vec(ϕ))
# end

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

