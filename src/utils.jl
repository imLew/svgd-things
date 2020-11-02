using Plots
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

function plot_known_dists(initial_dist, target_dist, alg_params, 
                      H₀, logZ, EV, dKL, q)
    # caption="""n_particles=$n_particles; n_iter=$n_iter; 
    #         norm_method=$norm_method; kernel_width=$kernel_width; 
    #         step_size=$step_size"""
    caption = ""
    caption_plot = plot(grid=false,annotation=(0.5,0.5,caption),
                      ticks=([]),fgborder=:white, subplot=1, framestyle=:none);
    # title = """$(typeof(initial_dist)) $(Distributions.params(initial_dist)) 
    #          target $(typeof(target_dist)) 
    #          $(Distributions.params(target_dist))"""
    title = ""
    title_plot = plot(grid=false,annotation=(0.5,0.5,title),
                      ticks=([]),fgborder=:white,subplot=1, framestyle=:none);
    int_plot, norm_plot = plot_integration(H₀, logZ, EV, dKL, 
                                           alg_params[:step_size])

    dim = size(q)[1]
    if dim > 3 
        layout = @layout [t{0.1h} ; d{0.3w} i ; c{0.1h}]
        display(plot(title_plot, norm_plot, int_plot, 
                      caption_plot, layout=layout, size=(1400,800), 
                      legend=:topleft));
    else
        if dim == 1
            dist_plot = plot_1D(initial_dist, target_dist, q)
        elseif dim == 2
            dist_plot = plot_2D(initial_dist, target_dist, q)
        # elseif dim == 3
        #     dist_plot = plot_3D(initial_dist, target_dist, q)
        end
    layout = @layout [t{0.1h} ; i ; a b; c{0.1h}]
    plot(title_plot, int_plot, norm_plot, dist_plot, 
         caption_plot, layout=layout, size=(1400,800), 
         legend=:topleft);
    end
end

function plot_2D(initial_dist, target_dist, q)
    # TODO add 'show title' parameter?
    dist_plot = scatter(q[1,:], q[2,:], 
                        labels="q");
    min_q = minimum(q)
    max_q = maximum(q)
    t = min_q-0.5*abs(min_q):0.05:max_q+0.5*abs(max_q)
    contour!(dist_plot, t, t, (x,y)->pdf(target_dist, [x, y]), color=:black, 
             label="p", levels=5)
    contour!(dist_plot, t, t, (x,y)->pdf(initial_dist, [x, y]), color=:black, 
             label="q_0", levels=5)
    return dist_plot
end

function plot_1D(initial_dist, target_dist, q)
    n_bins = length(q) ÷ 5
    dist_plot = histogram(reshape(q, length(q)), 
                          fillalpha=0.3, labels="q" ,bins=20,
                          normalize=true);
    min_q = minimum(q)
    max_q = maximum(q)
    t = min_q-0.2*abs(min_q):0.05:max_q+0.2*abs(max_q)
    plot!(x->pdf(target_dist, x), t, labels="p")
    plot!(x->pdf(initial_dist, x), t, labels="q₀")
    return dist_plot
end
