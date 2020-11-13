using BSON
using Distributions
using ValueHistories
using Plots

using SVGD

function data_filename(d)
    "$(d[:n_particles])particles_mu0=$(d[:μ₀])_S0=$(d[:Σ₀])_mup=$(d[:μₚ])_Sp=$(d[:Σₚ])_$(d[:n_iter])iter_stepsize=$(d[:step_size])"
end

function filter_by_key(key, values, data_array=all_data)
    out = []
    for d in data_array
        if d[key] ∈ values
            push!(out, d)
        end
    end
    return out
end

function filter_by_dict(dict, data_array=all_data)
    out = data_array
    for (k, v) in dict
        out = filter_by_key(k, v, out)
    end
    return out 
end

function estimation_ratios(data, key)
    dKL_hist = data[:svgd_results][1][1]

    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])
    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist )
    
    estimate_logZ.(H₀, EV, data[:step_size]*cumsum(get(dKL_hist, key)[2]))./data[:true_logZ]
end

function plot_gaussian_results(data)
    dKL_hist = data[:svgd_results][1][1]
    final_particles = data[:svgd_results][1][end]

    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])

    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist )
    true_logZ = logZ(target_dist)

    ## create a plot that contains the caption
    caption = "N($(data[:μ₀]), $(data[:Σ₀])) → N($(data[:μₚ]), $(data[:Σₚ]))"
    caption_plot = plot(grid=false, annotation=(0.5, 0.5, caption), 
                      ticks=([]), fgborder=:white, subplot=1, framestyle=:none);

    ## get plots for the different estimation methods
    # title of this plot is the title for the whole figure
    int_plot = plot( title="""$(data[:n_particles]) particles
                            $(data[:n_iter]) iterations
                            step_size $(data[:step_size])""" 
                    );
    hline!(int_plot, [true_logZ], labels="true log Z $(round(true_logZ, digits=3))", color=color[2]);
    for k in keys(dKL_hist)
        if k != :ϕ_norm  && k != :dKL_unbiased && k != :dKL_stein_discrep
            if k == :dKL_rkhs
                label = "RKHS norm"
                color = colors[1]
            elseif k == :dKL_stein_discrep
                label = "Stein discrepancy"
                color = colors[3]
            elseif k == :dKL_unbiased
                label = "unbiased Stein discrepancy"
                color = colors[4]
            end
            est_logZ = estimate_logZ.([H₀], [EV], 
                                    data[:step_size]*cumsum(get(dKL_hist, k)[2])
                                   )

            plot!(int_plot, est_logZ, labels=label, color=color)
        end
    end

    ## plot initial, target and variational distributions
    dist_plot = plot_2D_gaussian(initial_dist, target_dist, final_particles)
    
    norm_plot = plot(data[:svgd_results][1][1][:ϕ_norm])
    ## combine all plots
    layout = @layout [ i ; n b; c{0.1h}]
    plot(int_plot, norm_plot, dist_plot, caption_plot, 
         layout=layout, 
         legend=:bottomright,
         size=(1250,1250));
end

function plot_2D_gaussian(initial_dist, target_dist, q)
    dist_plot = scatter(q[1,:], q[2,:], legend=false, label="", msw=0.0, alpha=0.5, color=colors[1]);
    # get range to cover both distributions and the particles
    min_x = minimum([
                    q[1] - 0.5 * abs(minimum(q[1])), 
                    params(initial_dist)[1][1] - 3*params(initial_dist)[2][1,1], 
                    params(target_dist)[1][1] - 3*params(target_dist)[2][1,1]
                   ])
    max_x = maximum([
                    q[1] + 0.5 * abs(maximum(q[1])), 
                    params(initial_dist)[1][1] + 3*params(initial_dist)[2][1,1], 
                    params(target_dist)[1][1] + 3*params(target_dist)[2][1,1]
                   ])
                   
    min_y = minimum([
                    q[2] - 0.5 * abs(minimum(q[2])), 
                    params(initial_dist)[1][2] - 3*params(initial_dist)[2][2,2], 
                    params(target_dist)[1][2] - 3*params(target_dist)[2][2,2]
                   ])
    max_y = maximum([
                    q[2] + 0.5 * abs(maximum(q[2])), 
                    params(initial_dist)[1][2] + 3*params(initial_dist)[2][2,2], 
                    params(target_dist)[1][2] + 3*params(target_dist)[2][2,2]
                   ])
    x = min_x:0.05:max_x
    y = min_y:0.05:max_y
    contour!(dist_plot, x, y, (x,y)->pdf(target_dist, [x, y]), color=colors[2], 
             label="", levels=5, msw=0.0, alpha=0.6)
    contour!(dist_plot, x, y, (x,y)->pdf(initial_dist, [x, y]), color=colors[1], 
             label="", levels=5, msw=0.0, alpha=0.6)
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

function plot_convergence(data, plt=nothing ;title="", rkhs=true, sd=false, sdu=false)
    if plt == nothing
        plt = plot(title=title, legend=:bottomright, xlabel="iterations")
    end
    if sd
        e_sd = estimation_ratios.(data, :dKL_stein_discrep)
        plot!(plt, mean(e_sd), ribbon=std(e_sd), label="Stein", color=colors[3], alpha=0.8)
    end
    if sdu
        e_sdu = estimation_ratios.(data, :dKL_unbiased)
        plot!(plt, mean(e_sdu), ribbon=std(e_sdu), label="Unbiased", color=colors[4], alpha=0.8)
    end
    if rkhs
        e_rkhs = estimation_ratios.(data, :dKL_rkhs)
        if !sdu && !sd
            label = ""
        else
            label = "RKHS"
        end
        plot!(plt, mean(e_rkhs), ribbon=std(e_rkhs), color=colors[1], label=label, alpha=0.8)
    end
    if !sdu && !sd
        label = ""
    else
        label = "true value"
    end
    hline!(plt, [1], label=label, color=colors[2])
end

all_data = [BSON.load(n) for n in readdir("data/gaussian_to_gaussian", join=true)]
# for some reason the last file has multiple values for the estimates
pop!(all_data) 

# converged = []
# not_converged = []
# for d in all_data
#     m =  mean(get(d[:svgd_results][1][1], :ϕ_norm)[2][end-50:end]) 
#     if m < 0.05
#         push!(converged, d)
#     else
#         push!(not_converged, d)
#     end
# end

data_set = all_data

using ColorSchemes
colors = ColorSchemes.seaborn_colorblind

root_dest = "/home/lew/Documents/BCCN_Master/Stein_labrot/SteinIntegration_AABI/plots/"

# # data with 5000 iter by particl
# # use one target 
# # one step size
# p1 = plot(size=(175,175), ylims=(0.5,1.7), xticks=0:2500:5000, xlabel="iterations", ylabel="normalized log Z");
# plot_convergence( filter_by_dict( Dict(:n_particles => [50], :n_iter => [5000]), data_set), p1);
# savefig(dest*"50part.png")
# p1 = plot(size=(175,175), ylims=(0.5,1.7), xticks=0:2500:5000, xlabel="iterations");
# plot_convergence( filter_by_dict( Dict(:n_particles => [100], :n_iter => [5000]), data_set), p1);
# savefig(dest*"100part.png")
# p1 = plot(size=(175,175), ylims=(0.5,1.7), xticks=0:2500:5000, xlabel="iterations");
# # p1 = plot(title="100 particles");
# plot_convergence( filter_by_dict( Dict(:n_particles => [200], :n_iter => [5000]), data_set), p1);
# savefig(dest*"200part.png")
# p1 = plot(size=(175,175), ylims=(0.5,1.7), xticks=0:2500:5000, xlabel="iterations");
# # p1 = plot(title="500 particles");
# plot_convergence( filter_by_dict( Dict(:n_particles => [500], :n_iter => [5000]), data_set), p1);
# savefig(dest*"500part.png")
# p1 = plot(size=(175,175), ylims=(0.5,1.7), xticks=0:2500:5000, xlabel="iterations");
# # p1 = plot(title="1000 particles");
# plot_convergence( filter_by_dict( Dict(:n_particles => [1000], :n_iter => [5000]), data_set), p1);
# savefig(dest*"1000part.png")

# for Figure 1
# comparing different step sizes for all estimators
dest = root_dest*"by_stepsize/"

data_set = filter_by_dict(Dict(
                               :μ₀ => [[0.0, 0.0]],
                               :Σ₀ => [[1.0 0.0; 0.0 1.0]],
                               :n_particles => [200, 500],
                               :μₚ => [[4.0, 5.0]],
                               :Σₚ => [[1.0 0.5; 0.5 1.0]],
                               :n_iter => [5000]
                              ))
for data in data_set
    savefig(
            plot_all_estimators(data, legend=:none),
            dest*data_filename(data)*".png"
           )
    sleep(0.1)
end

function plot_all_estimators(data; size=(275,275), legend=:bottomright, ylims=(-Inf,Inf), lw=3)
    dKL_hist = data[:svgd_results][1][1]
    final_particles = data[:svgd_results][1][end]

    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])

    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist )
    true_logZ = logZ(target_dist)

    ## get plots for the different estimation methods
    # title of this plot is the title for the whole figure
    int_plot = plot(xlabel="iterations", ylabel="log Z", legend=legend, size=size, lw=lw);
    est_logZ = estimate_logZ.([H₀], [EV], data[:step_size]*cumsum(get(dKL_hist, :dKL_stein_discrep)[2]));
    plot!(int_plot, est_logZ, label="Stein", color=colors[3], lw=lw, ls=:dash);
    est_logZ = estimate_logZ.([H₀], [EV], data[:step_size]*cumsum(get(dKL_hist, :dKL_unbiased)[2]));
    plot!(int_plot, est_logZ, label="Unbiased", color=colors[4], lw=lw, ls=:dashdot);
    est_logZ = estimate_logZ.([H₀], [EV], data[:step_size]*cumsum(get(dKL_hist, :dKL_rkhs)[2]));
    plot!(int_plot, est_logZ, label="RKHS", color=colors[1], lw=lw, ls=:dot);
    hline!(int_plot, [true_logZ],ylims=ylims, color=colors[2], label="true value", lw=lw);
    return int_plot
end

# # create plots for individual runs
# for some reason this fails if we loop over n so the easiest is to just change 
# it to 1, 2 and 3 and run the last bit of code
n=3
    data = candidates[n]

    dKL_hist = data[:svgd_results][1][1]
    final_particles = data[:svgd_results][1][end]

    initial_dist = MvNormal(data[:μ₀], data[:Σ₀])
    target_dist = MvNormal(data[:μₚ], data[:Σₚ])

    H₀ = Distributions.entropy(initial_dist)
    EV = expectation_V( initial_dist, target_dist )
    true_logZ = logZ(target_dist)

    ## get plots for the different estimation methods
    # title of this plot is the title for the whole figure
    int_plot = plot(xlabel="iterations", ylabel="log Z");
    est_logZ = estimate_logZ.([H₀], [EV], data[:step_size]*cumsum(get(dKL_hist, :dKL_rkhs)[2]))
    plot!(int_plot, est_logZ, label="", color=colors[1]);
    hline!(int_plot, [true_logZ], labels="",ylims=(0.25*true_logZ,1.75*true_logZ), color=colors[2], ls=:dash);

    ## plot initial, target and variational distributions
    dist_plot = plot_2D_gaussian(initial_dist, target_dist, final_particles);

    if n != 3
        xticks=0:2500:5000
    else
        xticks=0:5000:10000
    end
    norm_plot = plot(data[:svgd_results][1][1][:ϕ_norm],ylims=(0,Inf),
                     markeralpha=0, label="", title="", xticks=xticks, color=colors[1],
                    xlabel="iterations", ylabel="||φ||");
    ## combine all plots
    layout = @layout [ i ; n b]

    final_plot = plot(int_plot, norm_plot, dist_plot, layout=layout, legend=:bottomright, size=(275,275))

    savefig(final_plot, dest*"gauss$n.png")
    data
