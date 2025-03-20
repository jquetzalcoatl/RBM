using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures, JLD2
using CUDA

include("utils/train.jl")
include("scripts/RAIS.jl")
include("configs/yaml_loader.jl")
config, _ = load_yaml_iter();

if config.gpu["gpu_bool"]
    dev = gpu
    CUDA.device_reset!()
    CUDA.device!(config.gpu["gpu_id"])
else
    dev = cpu
end

function estimate_LL(PATH::String,  modelname::String, v_test::Matrix{Float32}, p::Dict{Any,Any})
    
    s = size(readdir("$(PATH)/models/$(modelname)/J"),1)
    l=p["num_models"]
    ais = Vector{Float64}(undef, 0)
    rais = Vector{Float64}(undef, 0)
    num = Vector{Float64}(undef, 0)
    LL_R = Vector{Float64}(undef, 0)
    LL_A = Vector{Float64}(undef, 0)

    Δidx = s >= l ? Int(floor(s/l)) : 1
    for i in 1:min(l,s)
        idx = Δidx*i
        
        rbm, J, m, hparams, opt = loadModel(modelname, dev, idx=idx);
        
        push!(ais, AIS(J, hparams, p["num_chains"], p["BGS"], p["nbetas"], dev))
        push!(rais, RAIS(J, hparams, p["num_chains"], p["BGS"], p["nbetas"], dev) )
        push!(num, LL_numerator(v_test,J))
        push!(LL_R, num[end] - rais[end])
        push!(LL_A, num[end] - ais[end])
    end

    return ais, rais, num, LL_R, LL_A
end

function plot_and_save(modelname::String, PATH::String, ais, rais, num, LL_R, LL_A)

    isdir("$(PATH)/Figs/$(modelname)") || mkpath("$(PATH)/Figs/$(modelname)")

    f = plot( ais, xscale=:identity, color=:blue, label="AIS", markershape=:circle)
    f = plot!( rais, color=:black, label="reverse AIS", s=:auto, markershapes = :square, lw=0, markerstrokewidth=0)
    f = plot!(size=(700,500), xlabel="Epochs", frame=:box, ylabel="log(Z)", margin = 15mm)
    
    savefig(f, "$(PATH)/Figs/$(modelname)/ais_and_rais_$(modelname).png")
    
    f = plot( LL_A, xscale=:identity, color=:blue, label="loglikelihood AIS", markershape=:circle)
    f = plot!( LL_R, xscale=:identity, color=:magenta, label="loglikelihood RAIS", markershape=:square)
    f = plot!(size=(700,500), xlabel="Epochs ", frame=:box, ylabel="LL", margin = 15mm)
    
    savefig(f, "$(PATH)/Figs/$(modelname)/loglikelihood_ais_rais_$(modelname).png")
    
    jldsave("$(PATH)/Figs/$(modelname)/partition_analytics.jld", rais=rais, ais=ais, num=num, llr=LL_R, lla=LL_A)
    
end

if abspath(PROGRAM_FILE) == @__FILE__
    ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"]="2GiB"
    rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    x_i, _ = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);

    PATH = "/home/javier/Projects/RBM/Results/"
    
    p=config.loglikelihood
    
    for modelname in config.model_analysis["files"]
        @info modelname
        ais, rais, num, LL_R, LL_A = estimate_LL(PATH, modelname, x_i, p);
        plot_and_save(modelname, PATH, ais, rais, num, LL_R, LL_A)
    end
end