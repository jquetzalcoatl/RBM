using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures, JLD2, StatsBase
using CUDA

include("utils/train.jl")
include("scripts/RAIS.jl")
include("scripts/symTools.jl")
include("configs/yaml_loader.jl")
config, _ = load_yaml_iter();

function measure_div(modelname::String)
    if modelname==""
        rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    else
        rbm, J, m, hparams, opt = loadModel(modelname, dev);
    end
    F = LinearAlgebra.svd(J.w, full=true);
    
    MrotU = Matrix{Float64}(I, hparams.nv, hparams.nv)
    co = 0
    for j in 1:Int(floor(hparams.nv*0.1))
        @info "\t" j
        indxs = vcat(collect(1:co),shuffle(collect(co+1:hparams.nv))[1:hparams.nv-(2+co)])
        MrotU = MrotU*rotmnd_optimized(Diagonal(ones(hparams.nv))[:,indxs],2π*rand())
    end
    MrotV = Matrix{Float64}(I, hparams.nh, hparams.nh)
    co = 0
    for j in 1:Int(floor(hparams.nh*0.1))
        @info "\t" j
        indxs = vcat(collect(1:co),shuffle(collect(co+1:hparams.nh))[1:hparams.nh-(2+co)])
        MrotV = MrotV*rotmnd_optimized(Diagonal(ones(hparams.nh))[:,indxs],2π*rand())
    end
    Σ = cat(cpu(Diagonal(F.S)), (hparams.nv - hparams.nh > 0 ? zeros(abs(hparams.nv - hparams.nh), hparams.nh) : zeros(hparams.nv, abs(hparams.nv - hparams.nh))), 
        dims=(hparams.nv - hparams.nh > 0 ? 1 : 2) )
    w = reshape(cpu(J.w),:)
    w_rot = cpu(reshape(MrotU*cpu(F.U)*MrotU'*Σ*MrotV*cpu(F.Vt)*MrotV,:))
    

    p = fit(Histogram, w, nbins=1000)
    q = fit(Histogram, w_rot, p.edges[1], closed=:left)
    p_dis = p.weights/sum(p.weights)
    q_dis = q.weights/sum(q.weights)
    kl,js = div_(p_dis,q_dis)
    return w,w_rot, p_dis,q_dis,kl,js
end

function plot_and_save(modelname::String, PATH::String, w,w_rot, p_dis,q_dis,kl,js)
    fig = plot(w, st=:histogram, lw=0, normalize=true, 
         color=:magenta, label="Before rotation")
    fig = plot!(w_rot, st=:histogram, color=:blue, lw=0, normalize=true, 
        alpha=0.5, label="After rotation")
    fig = plot!(xlabel="Weights", ylabel="PDF", 
        tickfontsize=15, labelfontsize=15, legendfontsize=15, 
        frame=:box, size=(700,500), left_margin=3mm, bottom_margin=2mm, legend = :topleft)

    savefig(fig, "$(PATH)/Figs/$(modelname)/weight_mat_R_$(modelname).png")

    fig = plot(p_dis, color=:purple, label="Before rotation binned")
    fig = plot!(q_dis, color=:cyan, label="After rotation binned")
    fig = plot!(xlabel="Weights", ylabel="PDF", 
        tickfontsize=15, labelfontsize=15, legendfontsize=15, 
        frame=:box, size=(700,500), left_margin=3mm, bottom_margin=2mm, legend = :topleft)

    savefig(fig, "$(PATH)/Figs/$(modelname)/weight_mat_R_binned$(modelname).png")

    jldsave("$(PATH)/Figs/$(modelname)/divergence.jld", kl=kl, js=js, w=w, w_rot=w_rot)
end

if abspath(PROGRAM_FILE) == @__FILE__
    ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"]="2GiB"
    if config.gpu["gpu_bool"]
        dev = gpu
        CUDA.device_reset!()
        CUDA.device!(config.gpu["gpu_id"])
    else
        dev = cpu
    end
    rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    x_i, _ = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);

    PATH = "/home/javier/Projects/RBM/Results/"
    
    p=config.loglikelihood
    
    for modelname in config.model_analysis["files"]
        @info modelname
        w,w_rot, p_dis,q_dis,kl,js = measure_div(modelname);
        plot_and_save(modelname, PATH, w,w_rot, p_dis,q_dis,kl,js)
    end
end