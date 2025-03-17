using CUDA, JLD2, Flux
using Plots.PlotMeasures
# import MLDatasets
using RestrictedBoltzmannMachines
using CudaRBMs
using RestrictedBoltzmannMachines: Binary, BinaryRBM, initialize!, pcd!,
    aise, raise, logmeanexp, logstdexp, sample_v_from_v, log_pseudolikelihood

include("utils/train.jl")
include("../configs/yaml_loader.jl")
config, _ = load_yaml_iter();
if config.phase_diagrams["gpu_bool"]
    dev = gpu
else
    dev = cpu
end

function loadLandscapes(PATH = "/home/javier/Projects/RBM/Results/",  modelname = "PCD-500-replica2"; l=100, nbetas = 10_000)
    s = size(readdir("$(PATH)/models/$(modelname)/J"),1)
    nsamples=100
    R_ais = Vector{Float64}[]
    R_rev = Vector{Float64}[]
    LL = Vector{Float64}[]

    # nv=28*28, nh=500,

    Δidx = s >= l ? Int(floor(s/l)) : 1
    for i in 1:min(l,s)
        idx = Δidx*i
        # @info idx
        
        # J = load("$(PATH)/models/$(modelname)/J/J_$(idx).jld", "J")
        rbm, J, m, hparams, opt = loadModel(modelname, dev, idx=idx);
        J.w = gpu(J.w)
        J.b = gpu(J.b)
        J.a = gpu(J.a)
        
        rbm = RestrictedBoltzmannMachines.RBM(Binary(; θ=reshape(J.a,28,28)), Binary(; θ=J.b), reshape(J.w, 28,28,hparams.nh))
        v = train_x[:, :, rand(1:size(train_x, 3), 1000)] |> gpu
        v = sample_v_from_v(rbm, v; steps=1000);

        init = initialize!(Binary(; θ = zero(rbm.visible.θ)), v)
        
        push!(R_ais, aise(rbm; nbetas, nsamples, init) )
        push!(R_rev, raise(rbm; nbetas, init, v=v[:,:,rand(1:size(v, 3), nsamples)]) )
        push!(LL, mean(log_pseudolikelihood(CudaRBMs.cpu(rbm), train_x)))
        
    end

    return R_ais, R_rev, LL
end

function saveStuff(R_ais, R_rev, LL, modelname)
    isdir("$(PATH)/Figs/$(modelname)") || mkpath("$(PATH)/Figs/$(modelname)")
    f = plot( -mean.(R_ais), ribbon=std.(R_ais), xscale=:identity, color=:blue, label="AIS", markershape=:circle)
    f = plot!( -mean.(R_rev), ribbon=std.(R_rev), color=:black, label="reverse AIS", s=:auto, markershapes = :square, lw=0, markerstrokewidth=0)
    f = plot!(size=(700,500), xlabel="Epochs", frame=:box, ylabel="-log(Z)", margin = 15mm)
    
    savefig(f, "$(PATH)/Figs/$(modelname)/log_partition_$(modelname).png")
    
    f = plot( LL, xscale=:identity, color=:blue, label="pseudolikelihood", markershape=:circle)
    f = plot!(size=(700,500), xlabel="Epochs (x10)", frame=:box, ylabel="log(PL)", margin = 15mm)
    
    savefig(f, "$(PATH)/Figs/$(modelname)/pseudolikelihood_$(modelname).png")
    
    jldsave("$(PATH)/Figs/$(modelname)/partition_cossio.jld", rais=R_ais, rrev=R_rev, ll=LL)
    
end

if abspath(PROGRAM_FILE) == @__FILE__

    PATH = "/home/javier/Projects/RBM/Results/"
    l=100
    nbetas=100_000
    # dev = gpu
    # β = 1.0
    Float = Float32
    train_x = MLDatasets.MNIST(split=:train)[:].features
    train_y = MLDatasets.MNIST(split=:train)[:].targets;
    train_x = Array{Float}(train_x[:, :, :] .> 0.5);

    for modelname in config.model_analysis["files"]
        @info modelname
        R_ais, R_rev, LL = loadLandscapes(PATH, modelname; l, nbetas);

        saveStuff(R_ais, R_rev, LL, modelname)
    end
end