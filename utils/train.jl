using CUDA
using Plots, Statistics, Dates

include("init.jl")
include("structs.jl")
include("adamOpt.jl")
include("loader.jl")
include("en.jl")
include("tools.jl")

# Training function
function train(dict ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false, annealing=false, β=1, learnType="Rdm", gpu_usage = false, t_samp=100, num=25, optType="SGD", numbers=[0,1], snapshot=50, savemodel=true, γ=0.001, logging=false, io=nothing)
    try
        Int(sqrt(num))
    catch
        @warn "num's root needs to be an integer."
        return 0,0,0,0,0
    end
    
    rbm, J, m, hparams, rbmZ = initModel(; nv, nh, batch_size, lr, γ, t, gpu_usage, optType)

    if optType=="Adam"
        opt = initOptW(hparams, J) 
    elseif optType=="SGD"
        opt = 0
    end
    dev = selectDev(hparams)
    
    x = loadData(; hparams, dsName="MNIST01", numbers)
    if learnType == "Rdm"
        x_Gibbs = [rand(size(x[1])...) for i in 1:size(x,1)]
    elseif learnType == "CD" || learnType == "PCD"
        x_Gibbs = x
    end
    
    if annealing
        β = 1.0
    end 

    for epoch in 1:epochs
        enEpoch, ΔwEpoch, ΔaEpoch, ΔbEpoch, ZEpoch = [], [], [], [], []
        
#         Threads.@threads 
        for i in eachindex(x)
            Δw, Δa, Δb = loss(rbm, J, x[i], x_Gibbs[i]; hparams, β, dev)
            if learnType == "PCD"
                x_Gibbs[i] = rbm.v |> cpu
            end

            updateJ!(J, Δw, Δa, Δb, opt; hparams)

            append!(enEpoch, en(rbm,J) |> cpu)
            append!(ZEpoch, sum(exp.(H(rbm, J)/(hparams.nv+hparams.nh))) |> cpu)
            append!(ΔwEpoch, mean(Δw) |> cpu)
            append!(ΔaEpoch, mean(Δa) |> cpu)
            append!(ΔbEpoch, mean(Δb) |> cpu)
        end
        
        append!(m.enList, mean(enEpoch)/(hparams.nv+hparams.nh))
        append!(m.enSDList, std(enEpoch)/(hparams.nv+hparams.nh))
        append!(m.enZList, genEnZSample(rbmZ, J, hparams, m; sampleSize = hparams.batch_size, t_samp, β, dev)/(hparams.nv+hparams.nh))
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
        append!(m.Z, mean(ZEpoch))

        @info string(now())[1:end-4], epoch, m.enList[end]/(hparams.nv+hparams.nh), m.ΔwList[end], m.ΔaList[end], m.ΔbList[end], β
        logging ? flush(io) : nothing
        if epoch % snapshot == 0 
            savemodel ? saveModel(rbm, J, m, hparams; opt, path = dict["msg"], baseDir = dict["bdir"]) : nothing
            genSample(rbm, J, hparams, m; num, β, t=t_samp, plotSample, epoch, dict, dev)         
        end
        if annealing
            β = β + 1/epochs
        end
        
        x = reshuffle(x; hparams)
        if learnType == "Rdm"
            x_Gibbs = [rand(size(x[1])...) for i in 1:size(x,1)]
        elseif learnType == "CD"
            x_Gibbs = x #reshuffle(x_Gibbs; hparams)
        elseif learnType == "PCD"
            x_Gibbs = reshuffle(x_Gibbs; hparams)
        end
        
    end
    rbm, J, m, hparams, opt
end

# Continue Training function
function continuetrain(dict, modelname ; epochs=50, plotSample=false, t_samp=100, num=40, snapshot=50)
    
    batch_size=dict["batchsize"]
    lr=dict["lr"]
    t=dict["gibbs"]
    annealing=dict["annealing"]
    β=dict["beta"]
    learnType=dict["pcd"]
    gpu_usage=dict["gpu"]
    optType=dict["opt"]
    numbers=dict["numbers"]
    epochsInit=dict["epochs"]

    if gpu_usage
        dev=gpu
    else
        dev=cpu
    end
    
    rbm, J, m, hparams, opt = loadModel(modelname, dev);
    rbmZ = genRBM(hparams)
    
    x = loadData(; hparams, dsName="MNIST01", numbers)
    if learnType == "Rdm"
        x_Gibbs = [rand(size(x[1])...) for i in 1:size(x,1)]
    elseif learnType == "CD" || learnType == "PCD"
        x_Gibbs = x
    end
    
    if annealing
        β = 1.0
    end 

    for epoch in epochsInit:epochs
        enEpoch, ΔwEpoch, ΔaEpoch, ΔbEpoch = [], [], [], []
        
#         Threads.@threads 
        for i in eachindex(x)
            Δw, Δa, Δb = loss(rbm, J, x[i], x_Gibbs[i]; hparams, β, dev)
            if learnType == "PCD"
                x_Gibbs[i] = rbm.v |> cpu
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
        if learnType == "Rdm"
            x_Gibbs = [rand(size(x[1])...) for i in 1:size(x,1)]
        else
            x_Gibbs = reshuffle(x_Gibbs; hparams)
        end
        x = reshuffle(x; hparams)
    end
    rbm, J, m, hparams, opt
end

function reshuffle(x; hparams)
    cat_state = cat(x..., dims=2)
    idx = randperm(size(cat_state,2))
    new_state = cat_state[:,idx]
    [new_state[:,i] for i in Iterators.partition(1:size(new_state,2), hparams.batch_size)]
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end