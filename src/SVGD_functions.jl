using ProgressMeter
using Statistics
using ValueHistories
using KernelFunctions
using LinearAlgebra
using Random
using Flux
using Zygote
using Distances
using PDMats

# export svgd_fit
# export median_trick
# export kernel_grad_matrix
# export empirical_RKHS_norm
# export compute_phi_norm
# export unbiased_stein_discrep
# export stein_discrep_biased

function KL_integral(hist, key=:dKL_rkhs)
    cumsum(get(hist, :step_sizes)[2] .* get(hist, key)[2])
end
export KL_integral

function median_trick(x)
    if size(x)[end] == 1
        return 1
    end
    d = Distances.pairwise(Euclidean(), x, dims=2)
    median(d)^2/log(size(x)[end])
end
export median_trick

grad(f,x,y) = gradient(f,x,y)[1]

function svgd_fit(q, grad_logp ;n_iter=100, step_size=1,
                           norm_method="standard", kernel_width=nothing,
                           n_particles=50)
    # opt = Descent(step_size)
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
        # @info "Step $i / $n_iter"
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
        # εϕ = Flux.Optimise.apply!(opt, q, ϕ)
        # q .+= εϕ

        ϕ = calculate_phi_vectorized(kernel, q, grad_logp)
        # phi = calculate_phi(kernel, q, grad_logp)
        if step_size isa Number
            ϵ = step_size
            while ϵ * maximum(norm(ϕ)) > 1
                ϵ /= 2
            end
            push!(hist, :step_sizes, i, ϵ)
            q .+= ϵ*ϕ
        elseif length(step_size) == n_iter
            q .+= step_size[i]*ϕ
            push!(hist, :step_sizes, i, step_size[i])
        else
            @error """Step size is neither a number nor an array (of the
                      correct length""" step_size
        end

        dKL_rkhs = compute_phi_norm(q, kernel, grad_logp, 
                                     norm_method="RKHS_norm", ϕ=ϕ)
        dKL_unbiased = compute_phi_norm(q, kernel, grad_logp, 
                                     norm_method="unbiased", ϕ=ϕ)
        dKL_stein_discrep = compute_phi_norm(q, kernel, grad_logp, 
                                     norm_method="standard", ϕ=ϕ)
        push!(hist, :dKL_unbiased, i, dKL_unbiased)
        push!(hist, :dKL_stein_discrep, i, dKL_stein_discrep)
        push!(hist, :dKL_rkhs, i, dKL_rkhs)
        push!(hist, :ϕ_norm, i, mean(norm(ϕ)))
        push!(hist, :Σ, i, cov(q, dims=2))
    end
    return q, hist
end
export svgd_fit

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
export calculate_phi

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
export calculate_phi_vectorized

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
export compute_phi_norm

function empirical_RKHS_norm(kernel::Kernel, q, ϕ)
    if size(q)[1] == 1
        invquad(kernelpdmat(kernel, q), vec(ϕ))
    else
        # this first method tries to flatten the tensor equation
        # invquad(flat_matrix_kernel_matrix(kernel, q), vec(ϕ))
        # the second method should be the straight forward case for a
        # kernel that is a scalar f(x) times identity matrix
        norm = 0
        try 
            k_mat = kernelpdmat(kernel, q)
            for f in eachrow(ϕ)
                norm += invquad(k_mat, vec(f))
            end
            return norm
        catch e
            if e isa PosDefException
                @show kernel
            end
            rethrow(e)
        end
    end
end
export empirical_RKHS_norm

# function empirical_RKHS_norm(kernel::MatrixKernel, q, ϕ)
# export empirical_RKHS_norm
#     invquad(flat_matrix_kernel_matrix(kernel, q), vec(ϕ))
# end

function unbiased_stein_discrep(q, kernel, grad_logp)
    n = size(q)[end]
    h = 1/kernel.transform.s[1]^2
    d = size(q)[1]
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    dKL = 0
    for (i, x) in enumerate(eachcol(q))
        glp_x = grad_logp(x)
        for (j, y) in enumerate(eachcol(q))
            if i != j
                dKL += k_mat[i,j] * dot(glp_x, grad_logp(y))
                dKL += dot( gradient(x->kernel(x,y), x)[1], grad_logp(y) )
                dKL += dot( gradient(y->kernel(x,y), y)[1], glp_x )
                # dKL += kernel(x,y) * ( 2d/h - 4/h^2 * SqEuclidean()(x,y))
            end
        end
    end
    dKL += sum(k_mat .* ( 2*d/h .- 4/h^2 * pairwise(SqEuclidean(), q)))
    dKL /= n*(n-1)
end
export unbiased_stein_discrep

function stein_discrep_biased(q, kernel, grad_logp) 
    n = size(q)[end]
    h = 1/kernel.transform.s[1]^2
    d = size(q)[1]
    k_mat = KernelFunctions.kernelmatrix(kernel, q)
    dKL = 0
    for (i, x) in enumerate(eachcol(q))
        glp_x = grad_logp(x)
        for (j, y) in enumerate(eachcol(q))
            dKL += k_mat[i,j] * dot(glp_x, grad_logp(y))
            dKL += dot( gradient(x->kernel(x,y), x)[1], grad_logp(y) )
            dKL += dot( gradient(y->kernel(x,y), y)[1], glp_x )
            # dKL += k_mat[i,j] * ( 2*d/h - 4/h^2 * SqEuclidean()(x,y))
        end
    end
    dKL += sum(k_mat .* ( 2*d/h .- 4/h^2 * pairwise(SqEuclidean(), q)))
    dKL /= n^2
end
export stein_discrep_biased

function kernel_grad_matrix(kernel::KernelFunctions.Kernel, q)
    if size(q)[end] == 1
        return 0
    end
	mapslices(x -> grad.(kernel, [x], eachcol(q)), q, dims = 1)
end
export kernel_grad_matrix

