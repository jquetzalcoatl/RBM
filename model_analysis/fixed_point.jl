using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures
using CUDA
CUDA.device_reset!()
CUDA.device!(1)
include("../utils/init.jl")
# include("../scripts/exact_partition.jl")
include("../scripts/PhaseAnalysis.jl")

# include("../configs/yaml_loader.jl")
# config, _ = load_yaml_iter();
# if config.phase_diagrams["gpu_bool"]
#     dev = gpu
# else
#     dev = cpu
# end
dev
config.model_analysis["files"]

modelName = config.model_analysis["files"][1]
modelName = "CD-500-T1-BW-replica1"

rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=100);
rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")

function get_us(J::Weights, hparams::HyperParams)
    F = LinearAlgebra.svd(J.w, full=true)
    ys = F.Vt * dev(ones(hparams.nh)) * 0.5
    xs = F.U' * dev(ones(hparams.nv)) * 0.5
    a = F.U' * J.a
    b = F.Vt * J.b
    x0 = hparams.nv > hparams.nh ? -b ./ F.S : -b[1:hparams.nv] ./ F.S
    y0 = hparams.nv > hparams.nh ? -a[1:hparams.nh] ./ F.S : -a ./ F.S

    if hparams.nv > hparams.nh
        us = 1/sqrt(2) .* (xs[1:hparams.nh] .+ ys .- x0 .- y0)
        ws = 1/sqrt(2) .* (-xs[1:hparams.nh] .+ ys .+ x0 .- y0)
        aa = a[1:hparams.nh]
        bb = b
    else
        us = 1/sqrt(2) .* (xs .+ ys[1:hparams.nv] .- x0 .- y0)
        ws = 1/sqrt(2) .* (-xs .+ ys[1:hparams.nv] .+ x0 .- y0)
        aa = a
        bb = b[1:hparams.nv]
    end
    us, ws, F.S, aa, bb, x0, y0
end

function get_us(v, h, J::Weights, hparams::HyperParams)
    F = LinearAlgebra.svd(J.w, full=true)
    ys = F.Vt * h
    xs = F.U' * v
    a = F.U' * J.a
    b = F.Vt * J.b
    x0 = hparams.nv > hparams.nh ? -b ./ F.S : -b[1:hparams.nv] ./ F.S
    y0 = hparams.nv > hparams.nh ? -a[1:hparams.nh] ./ F.S : -a ./ F.S

    if hparams.nv > hparams.nh
        us = 1/sqrt(2) .* (xs[1:hparams.nh] .+ ys .- x0 .- y0)
        ws = 1/sqrt(2) .* (-xs[1:hparams.nh] .+ ys .+ x0 .- y0)
        aa = a[1:hparams.nh]
        bb = b
    else
        us = 1/sqrt(2) .* (xs .+ ys[1:hparams.nv] .- x0 .- y0)
        ws = 1/sqrt(2) .* (-xs .+ ys[1:hparams.nv] .+ x0 .- y0)
        aa = a
        bb = b[1:hparams.nv]
    end
    us, ws, F.S, aa, bb, x0, y0
end

us, ws, Λ, a, b, x_sp, y_sp = get_us(J,hparams)
v,h = gibbs_sample(J,hparams,100,500)
us_gibbs, ws_gibbs, _, _, _, _, _ = get_us(v,h,J,hparams)

plot(cpu(us) .^ 2 ./ cpu(Λ), yscale=:log)
plot(abs.(cpu(us)) ./ cpu(sqrt.(Λ)))
plot!(abs.(mean(cpu(us_gibbs),dims=2)) ./ cpu(sqrt.(Λ)), ribbon=std(cpu(us_gibbs),dims=2))

plot(cpu(us) ./ cpu(sqrt.(Λ)), mean(cpu(us_gibbs) ,dims=2) ./ cpu(sqrt.(Λ)))
plot!(-30:30, x->x)

plot()
for i in 1:100
    plot!(cpu(us_gibbs)[i,:], st=:histogram, lw=0, opacity=0.5)
end
plot!()

plot(cpu(-(us .^ 2 .- ws .^ 2) .* Λ ./ 2)[2:100], lw=2, marker=:circle, markerstrokewidth=0)
plot!(cpu(-(mean(us_gibbs ,dims=2) .^ 2 .- mean(ws_gibbs ,dims=2) .^ 2) .* Λ ./ 2)[2:100], lw=2, marker=:circle, markerstrokewidth=0)
plot!(cpu(a .* b ./ Λ )[2:100], lw=2, marker=:circle, markerstrokewidth=0)
plot!(xscale=:log)


begin
    Array(transpose(v))
    lnum=Int(sqrt(100))
    mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
    mat_rot = reverse(transpose(mat), dims=1)
    f1 = heatmap(cpu(mat_rot), size=(900,900))
    # savefig(f1, PATH * "Sample_$modelName.png")
    f1
end

v,h = gibbs_sample(J,hparams,500,1)
us_BW, ws_BW, _, _, _ = get_us(v,h,J,hparams)
usBW = cpu(us_BW)
for i in 1:500
    v,h = gibbs_sample(v,J,hparams,1,500)
    us_BW, ws_BW, _, _, _ = get_us(v,h,J,hparams)
    usBW = cat(usBW,cpu(us_BW), dims=3)
end


# plot(usBW[1,:,:]', lw=2, marker=:circle, markerstrokewidth=0, label=false)
# plot(usBW[2,:,:]', lw=2, marker=:circle, markerstrokewidth=0, label=false)
us

begin
    idx=2
    @info cpu(Λ)[idx]
    plot(mean(usBW[idx,:,:], dims=1)', ribbon=std(usBW[idx,:,:], dims=1)', lw=2, marker=:circle, markerstrokewidth=0, label=false)
    hline!([mean(mean(usBW[idx,:,:], dims=1)[1,20:end])], linestyle=:dash, c=:black, lw=3, label="mean")

    hline!([cpu(x_sp)[idx]], lw=3, label="saddle point")
    hline!([cpu(us)[idx]], lw=3, label="mu")
end
