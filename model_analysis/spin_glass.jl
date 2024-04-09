begin
    using CUDA, Flux, HDF5
    using Base.Threads
    using StatsPlots
    using Plots.PlotMeasures
    CUDA.device_reset!()
    CUDA.device!(0)
    Threads.nthreads()
end

begin
    include("../therm.jl")
    include("../configs/yaml_loader.jl")
    PATH = "/home/javier/Projects/RBM/Results/"
    dev = gpu
    β = 1.0
    config, _ = load_yaml_iter();
end

begin
    modelName = config.model_analysis["files"][1]
    rbm, J, m, hparams, opt = loadModel(modelName, gpu);
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    # idx=100
    # J = load("$(PATH)/models/$(modelName)/J/J_$(idx).jld", "J")
    J.w = gpu(J.w)
    J.b = gpu(J.b)
    J.a = gpu(J.a)
    F = LinearAlgebra.svd(J.w, full=true);
    v_val,h_val, x_val,y_val = data_val_samples(F, J, x_i, y_i; hparams) #(F, avg=false)
    z_val = size(y_val,1) <= size(x_val,1) ? cat(y_val, x_val, dims=1) : cat(x_val, y_val, dims=1)
end


function genera_phase_diag(modelname, hparams)
    phase_diag = zeros(100,2)
    if contains(modelname, "PCD")
        ran = 1:100
    else
        ran = 1:10:1000
    end
    for (c,i) in enumerate(ran)
        J = load("$(PATH)/models/$(modelname)/J/J_$(i).jld", "J")
        J.w = gpu(J.w)
        J.b = gpu(J.b)
        J.a = gpu(J.a)
        # J0 = sum(J.w)/(hparams.nv + hparams.nh)
        # JJ0 = sqrt(sum((J.w .- gpu(J0)) .^ 2 ) ) /(hparams.nv + hparams.nh) .^ (1/2) 
        J0 = mean(J.w) * (hparams.nv + hparams.nh)
        JJ0 = std(J.w) * (hparams.nv + hparams.nh) .^ (1/2) 
        phase_diag[c,:] .= abs(J0)/JJ0, 1/JJ0
    end
    phase_diag
end

ph_diag_dict = Dict()
for n in 1:length(config.model_analysis["files"])
    # J, modelname = initializer(n)
    modelname = config.model_analysis["files"][n]
    rbm, J, m, hparams, opt = loadModel(modelname, gpu);
    ph_diag_dict[modelname] = genera_phase_diag(modelname, hparams)
end

begin
    plot()
    for modelname in sort(collect(keys(ph_diag_dict)))[1:5]
        plot!(ph_diag_dict[modelname][:,1], ph_diag_dict[modelname][:,2], s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.01, opacity=0.9, frame=:box, label=modelname)
    end
    for modelname in sort(collect(keys(ph_diag_dict)))[1+5:10]
        plot!(ph_diag_dict[modelname][:,1], ph_diag_dict[modelname][:,2], s=:auto, markershapes = :square, lw=0.0, markerstrokewidth=0.01, opacity=0.9, frame=:box, label=modelname)
    end
    for modelname in sort(collect(keys(ph_diag_dict)))[1+10:15]
        plot!(ph_diag_dict[modelname][:,1], ph_diag_dict[modelname][:,2], s=:auto, markershapes = :diamond, lw=0.0, markerstrokewidth=0.01, opacity=0.9, frame=:box, label=modelname)
    end
    for modelname in sort(collect(keys(ph_diag_dict)))[1+15:20]
        plot!(ph_diag_dict[modelname][:,1], ph_diag_dict[modelname][:,2], s=:auto, markershapes = :hexagon, lw=0.0, markerstrokewidth=0.01, opacity=0.9, frame=:box, label=modelname)
    end
    for modelname in sort(collect(keys(ph_diag_dict)))[1+20:25]
        plot!(ph_diag_dict[modelname][:,1], ph_diag_dict[modelname][:,2], s=:auto, markershapes = :star6, lw=0.0, markerstrokewidth=0.01, opacity=0.9, frame=:box, label=modelname)
    end
    plot!(legend=false, xlabel="J0/J", ylabel="kT/J", size=(900,800))
end
hline!([1])
vline!([1])
plot!([J0/JJ0], [1/JJ0], s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.1, frame=:box)

JJ = initWeights(hparams)
###################
# N*<J>
# N^1/2 * std(J)
# J0 = sum(-J.w)/(hparams.nv + hparams.nh)
# JJ0 = sqrt(sum((-J.w .- gpu(J0)) .^ 2 ) ) /(hparams.nv + hparams.nh) .^ (1/2) 
# @info J0/JJ0, 1/JJ0

J0 = mean(JJ.w) * (hparams.nv + hparams.nh)
JJ0 =  std(JJ.w) * (hparams.nv + hparams.nh) .^ (1/2) 
@info abs(J0)/JJ0, 1/JJ0