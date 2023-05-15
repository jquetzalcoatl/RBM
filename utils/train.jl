using CUDA
using Plots, Statistics

include("init.jl")
include("structs.jl")
include("loader.jl")
include("en.jl")

# Training function
function train( ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false, annealing=false, β=1, PCD=true, gpu_usage = false, t_samp=100, optType="SGD")
    
    rbm, J, m, hparams = initModel(; nv, nh, batch_size, lr, t, gpu_usage, optType)
    dev = selectDev(hparams)
    x = loadData(; hparams, dsName="MNIST01")
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

            updateJ!(J, Δw, Δa, Δb; hparams)

            append!(enEpoch, en(rbm,J) |> cpu)
            append!(ΔwEpoch, mean(Δw) |> cpu)
            append!(ΔaEpoch, mean(Δa) |> cpu)
            append!(ΔbEpoch, mean(Δb) |> cpu)
        end
        append!(m.enList, mean(enEpoch)/(hparams.nv+hparams.nh))
        append!(m.enSDList, std(enEpoch)/(hparams.nv+hparams.nh))
        append!(m.ΔwList, mean(ΔwEpoch))
        append!(m.ΔwSDList, std(ΔwEpoch))
        append!(m.ΔaList, mean(ΔaEpoch))
        append!(m.ΔaSDList, std(ΔaEpoch))
        append!(m.ΔbList, mean(ΔbEpoch))
        append!(m.ΔbSDList, std(ΔbEpoch))
        if epoch % 1 == 0
            @info epoch, m.enList[end]/(hparams.nv+hparams.nh), m.ΔwList[end], m.ΔaList[end], m.ΔbList[end], β
            if plotSample
                genSample(rbm, J, hparams, m; num = 4, β, t=t_samp, dev)
            end
        end
        if annealing
            β = β + 1/epochs
        end
        if PCD
            PCD_state = reshuffle(PCD_state; hparams)
        end
    end
    rbm, J, m, hparams, 0
end

function trainAdam( ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false, annealing=false, β=1, PCD=true, gpu_usage = false, t_samp=100, num=40, optType="Adam")
    
    rbm, J, m, hparams = initModel(; nv, nh, batch_size, lr, t, gpu_usage, optType)
    opt = initOptW(hparams, J) 
    dev = selectDev(hparams)
    x = loadData(; hparams, dsName="MNIST01")
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

#             updateJ!(J, Δw, Δa, Δb; hparams)
            updateJAdam!(J, Δw, Δa, Δb, opt; hparams)

            append!(enEpoch, en(rbm,J) |> cpu)
            append!(ΔwEpoch, mean(Δw) |> cpu)
            append!(ΔaEpoch, mean(Δa) |> cpu)
            append!(ΔbEpoch, mean(Δb) |> cpu)
        end
        append!(m.enList, mean(enEpoch)/(hparams.nv+hparams.nh))
        append!(m.enSDList, std(enEpoch)/(hparams.nv+hparams.nh))
        append!(m.ΔwList, mean(ΔwEpoch))
        append!(m.ΔwSDList, std(ΔwEpoch))
        append!(m.ΔaList, mean(ΔaEpoch))
        append!(m.ΔaSDList, std(ΔaEpoch))
        append!(m.ΔbList, mean(ΔbEpoch))
        append!(m.ΔbSDList, std(ΔbEpoch))
        if epoch % 1 == 0
            @info epoch, m.enList[end], m.enSDList[end], m.ΔwList[end], m.ΔaList[end], m.ΔbList[end], β
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