using Distributions
using Random
using KernelFunctions
using LinearAlgebra
using Zygote
using ForwardDiff
using PDMats

export gradp
export kernel_gradient
export flat_matrix_kernel_matrix

flatten_index(i, j, j_max) = j + j_max *(i-1)

function flatten_tensor(K)
    d_max, l_max, i_max, j_max = size(K)
    K_flat = Matrix{Float64}(undef, d_max*i_max, l_max*j_max)
    for d in 1:d_max
        for l in 1:l_max
            for i in 1:i_max
                for j in 1:j_max
                    K_flat[ flatten_index(d, i, i_max), 
                            flatten_index(l, j, j_max) ] = K[d,l,i,j]
                end
            end
        end
    end
    return K_flat
end

# struct MatrixKernel <: KernelFunctions.Kernel end

function flat_matrix_kernel_matrix(k::Kernel, q)
    d, n = size(q)
    # kmat = Array{Float64}{undef, d, d, n, n}
    kmat = zeros(d,d,n,n)
    for (i, x) in enumerate(eachcol(q))
        for (j, y) in enumerate(eachcol(q))
            kmat[:,:,i,j] = k(x,y) .* I(d)
        end
    end
    get_pdmat(flatten_tensor(kmat))
end

function get_pdmat(K)
    Kmax =maximum(K)
    α = eps(eltype(K))
    while !isposdef(K+α*I) && α < 0.01*Kmax
        α *= 2.0
    end
    if α >= 0.01*Kmax
        throw(ErrorException("""Adding noise on the diagonal was not 
                             sufficient to build a positive-definite 
                             matrix:\n\t- Check that your kernel parameters 
                             are not extreme\n\t- Check that your data is 
                             sufficiently sparse\n\t- Maybe use a different 
                             kernel"""))
    end
    return PDMat(K+α*I)
end

function gradp(d::Distribution, x)
    if length(x) == 1
        g = Zygote.gradient(x->log(pdf.(d, x)[1]), x )[1]
        if g == nothing
            @info "x" x
            println("gradient nothing")
            g = 0
        end
        return g
    end
    ForwardDiff.gradient(x->log(pdf(d, x)), reshape(x, length(x)) )
end

function kernel_gradient(k::Kernel, x, y)
    Zygote.gradient( x->k(x,y), x)[1]
end

# function kernel_gradient(k::TransformedKernel{SqExponentialKernel},x,y)
#     h = 1/k.transform.s[1]^2
#     -2/h * (x-y) * exp(-h\norm(x-y))
# end

# bayesian logistic regression
# regression util functions
function plot_classes(data, truth)
    # only 2D 
    x = traindata[:,2]
    y = traindata[:,3]
    l = traindata[:,1]
    t = data[:,1]
    t.!=y
    Plots.scatter([x[l.==t], x[l.!=t]], [y[l.==t], y[l.!=t]], 
                  label=["correct" "incorrect"])
    #TODO: color by average of prediction
end

