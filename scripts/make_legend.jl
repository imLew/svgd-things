using Plots; pyplot()
using ColorSchemes: seaborn_colorblind
c = seaborn_colorblind

rkhsc = c[1]
steinc = c[3]
usteinc = c[4]
truthc = c[2]

rkhss = :dot
steins = :dash
usteins = :dashdot
truths = :solid

p = plot([[],[],[],[]], [[],[],[],[]],
    color = [rkhsc steinc usteinc truthc],
    linestyle = [rkhss steins usteins truths],
    label = ["RKHS" "Stein" "Unbiased Stein" "Truth"],
    lw = 20.0,
    showaxis = false,
    framestyle=:none,
    legend = :left,
    legendfontsize = 30.0,
    foreground_color_legend = RGBA(0,0,0,0),
    background_color_legend = RGBA(0,0,0,0),
)

display(p)
savefig("~/Tex Projects/SteinIntegration_AABI/plots/by_stepsize/legend.png")