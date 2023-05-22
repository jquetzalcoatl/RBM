using CUDA
using Plots, Statistics, Dates

include("init.jl")
include("structs.jl")
include("adamOpt.jl")
include("loader.jl")
include("en.jl")
include("tools.jl")

# Training function
function train(dict ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false, annealing=false, β=1, PCD=true, gpu_usage = false, t_samp=100, num=40, optType="SGD", numbers=[0,1], snapshot=50)
    
    rbm, J, m, hparams, rbmZ = initModel(; nv, nh, batch_size, lr, t, gpu_usage, optType)

    if optType=="Adam"
        opt = initOptW(hparams, J) 
    elseif optType=="SGD"
        opt = 0
    end
    dev = selectDev(hparams)
    
    x = loadData(; hparams, dsName="MNIST01", numbers)
    PCD_state = x
    if annealing
        β = 1.0
    end 

    for epoch in 1:epochs
        enEpoch, ΔwEpoch, ΔaEpoch, ΔbEpoch = [], [], [], []
        
#         Threads.@threads 
        for i in eachindex(x)
            Δw, Δa, Δb = loss(rbm, J, PCD_state[i]; hparams, β, dev)
            if PCD
                PCD_state[i] = rbm.v |> cpu
            end

            updateJ!(J, Δw, Δa, Δb, opt; hparams)

            append!(enEpoch, en(rbm,J) |> cpu)
            append!(ΔwEpoch, mean(Δw) |> cpu)
            append!(ΔaEpoch, mean(Δa) |> cpu)
            append!(ΔbEpoch, mean(Δb) |> cpu)
        end
        
        append!(m.enList, mean(enEpoch)/(hparams.nv+hparams.nh))
        append!(m.enSDList, std(enEpoch)/(hparams.nv+hparams.nh))
        append!(m.enZList, genEnZSample(rbmZ, J, hparams, m; sampleSize = 1000, t_samp, β, dev)/(hparams.nv+hparams.nh))
        append!(m.ΔwList, mean(ΔwEpoch))
        append!(m.ΔwSDList, std(ΔwEpoch))
        append!(m.ΔaList, mean(ΔaEpoch))
        append!(m.ΔaSDList, std(ΔaEpoch))
        append!(m.ΔbList, mean(ΔbEpoch))
        append!(m.ΔbSDList, std(ΔbEpoch))
        append!(m.wMean, MatrixMean(J.w))
        append!(m.wVar, MatrixVar(J.w))
        append!(m.wTrMean, MatrixMean(J.w'))
        append!(m.wTrVar, MatrixVar(J.w'))
        
        if epoch % snapshot == 0
            @info now(), epoch, m.enList[end]/(hparams.nv+hparams.nh), m.ΔwList[end], m.ΔaList[end], m.ΔbList[end], β
            saveModel(rbm, J, m, hparams; opt, path = dict["msg"], baseDir = dict["bdir"])
            if plotSample
                genSample(rbm, J, hparams, m; num, β, t=t_samp, dev)
            end
        end
        if annealing
            β = β + 1/epochs
        end
        if PCD
            PCD_state = reshuffle(PCD_state; hparams)
        end
    end
    rbm, J, m, hparams, opt
end

function reshuffle(PCD_state; hparams)
    cat_state = cat(PCD_state..., dims=2)
    idx = randperm(size(cat_state,2))
    new_state = cat_state[:,idx]
    [new_state[:,i] for i in Iterators.partition(1:size(new_state,2), hparams.batch_size)]
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end