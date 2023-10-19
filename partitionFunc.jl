using CUDA, Flux, JLD2
using Plots.PlotMeasures
import MLDatasets
using RestrictedBoltzmannMachines
# using Statistics: mean, std, middle
# using ValueHistories: MVHistory
using CudaRBMs
using RestrictedBoltzmannMachines: Binary, BinaryRBM, initialize!, pcd!,
    aise, raise, logmeanexp, logstdexp, sample_v_from_v, log_pseudolikelihood

include("utils/train.jl")

function loadLandscapes(PATH = "/home/javier/Projects/RBM/Results/",  modelname = "PCD-500-replica2"; l=30, nv=28*28, nh=500, nbetas = 10_000)
    s = size(readdir("$(PATH)/models/$(modelname)/J"),1)
    nsamples=100
    # ndists = [10, 100, 1000, 10_000, 100_000]
    # nbetas = 10_000
    R_ais = Vector{Float64}[]
    R_rev = Vector{Float64}[]
    LL = []

    Δidx = s >= l ? Int(floor(s/l)) : 1
    for i in 1:min(l,s)
        idx = Δidx*i
        # @info idx
        
        J = load("$(PATH)/models/$(modelname)/J/J_$(idx).jld", "J")
        J.w = gpu(J.w)
        J.b = gpu(J.b)
        J.a = gpu(J.a)
        
        rbm = RestrictedBoltzmannMachines.RBM(Binary(; θ=reshape(J.a,28,28)), Binary(; θ=J.b), reshape(J.w, 28,28,500))
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
    f = plot!(size=(700,500), xlabel="Epochs (x10)", frame=:box, ylabel="-log(Z)", margin = 15mm)
    
    savefig(f, "$(PATH)/Figs/$(modelname)/log_partition_$(modelname).png")
    
    f = plot( LL, xscale=:identity, color=:blue, label="pseudolikelihood", markershape=:circle)
    f = plot!(size=(700,500), xlabel="Epochs (x10)", frame=:box, ylabel="log(PL)", margin = 15mm)
    
    savefig(f, "$(PATH)/Figs/$(modelname)/pseudolikelihood_$(modelname).png")
    
    jldsave("$(PATH)/Figs/$(modelname)/partition_cossio.jld", rais=R_ais, rrev=R_rev, ll=LL)
    
end

PATH = "/home/javier/Projects/RBM/Results/"
l=100
nv=28*28
nh=500
nbetas=100_000
dev = gpu
β = 1.0
Float = Float32
train_x = MLDatasets.MNIST(split=:train)[:].features
train_y = MLDatasets.MNIST(split=:train)[:].targets;
train_x = Array{Float}(train_x[:, :, :] .> 0.5);

for model in ["Rdm-500-T10-BW-replica", "Rdm-500-T100-BW-replica", "CD-500-T1-replica", "CD-500-T1-BW-replica", "CD-500-T10-BW-replica", "CD-500-T100-BW-replica", "CD-500-T1000-5-BW-replica-L", "PCD-500-replica", "PCD-100-replica"]
    for i in 1:5
        # modelname = "Rdm-500-T1-replica$(i)"
        # modelname = "Rdm-500-T1-BW-replica$(i)"
        # modelname = "CD-500-T1-replica$(i)"
        # modelname = "CD-500-T1-BW-replica$(i)"
        # modelname = "CD-500-T10-BW-replica$(i)"
        # modelname = "Rdm-500-T10-BW-replica$(i)"
        # modelname = "CD-500-T100-BW-replica$(i)"
        # modelname = "Rdm-500-T100-BW-replica$(i)"

        # modelname = "CD-500-T1000-5-BW-replica$(i)-L"
        if model != "CD-500-T1000-5-BW-replica-L"
            modelname = model * "$(i)"
        else
            modelname = "CD-500-T1000-5-BW-replica$(i)-L"
        end
        # modelname = "PCD-500-replica$(i)"
        R_ais, R_rev, LL = loadLandscapes(PATH, modelname; l, nv, nh, nbetas);

        saveStuff(R_ais, R_rev, LL, modelname)
    end
end