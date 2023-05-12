using CUDA
using Plots, Statistics

include("init.jl")
include("structs.jl")
include("loader.jl")
include("en.jl")

# Training function
function train( ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false, annealing=false, β=1, PCD=true, gpu_usage = false, t_samp=100)
    rbm, J, m, hparams = initModel(; nv, nh, batch_size, lr, t, gpu_usage)
    dev = selectDev(hparams)
    x = loadData(; hparams, dsName="MNIST01")
    PCD_state = x
    if annealing
        β = 0
    end 

    for epoch in 1:epochs
        enEpoch, ΔwEpoch, ΔaEpoch, ΔbEpoch = 0, 0, 0, 0
        
#         Threads.@threads 
        for i in eachindex(x)
            Δw, Δa, Δb = loss(rbm, J, PCD_state[i]; hparams, β, dev)
            if PCD
                PCD_state[i] = rbm.v |> cpu
            end

            updateJ!(J, Δw, Δa, Δb; hparams)

            enEpoch = enEpoch + en(rbm,J) |> cpu
            ΔwEpoch = ΔwEpoch + mean(Δw) |> cpu
            ΔaEpoch = ΔaEpoch + mean(Δa) |> cpu
            ΔbEpoch = ΔbEpoch + mean(Δb) |> cpu
        end
        append!(m.enList, enEpoch/size(x,1))
        append!(m.ΔwList, ΔwEpoch/size(x,1))
        append!(m.ΔaList, ΔaEpoch/size(x,1))
        append!(m.ΔbList, ΔbEpoch/size(x,1))
        if epoch % 1 == 0
            @info epoch, m.enList[end], m.ΔwList[end], m.ΔaList[end], m.ΔbList[end], β
            if plotSample
                genSample(rbm, J, hparams, m; num = 4, t=t_samp, dev)
            end
        end
        if annealing
            β = β + 1/epochs
        end
        if PCD
            PCD_state = reshuffle(PCD_state; hparams)
        end
    end
    rbm, J, m, hparams
end

function trainAdam( ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false, annealing=false, β=1, PCD=true, gpu_usage = false, t_samp=100, num=40)
    
    rbm, J, m, hparams = initModel(; nv, nh, batch_size, lr, t, gpu_usage)
    opt = initOptW(hparams, J) 
    dev = selectDev(hparams)
    x = loadData(; hparams, dsName="MNIST01")
    PCD_state = x
    if annealing
        β = 0
    end 

    for epoch in 1:epochs
        enEpoch, ΔwEpoch, ΔaEpoch, ΔbEpoch = 0, 0, 0, 0
        
#         Threads.@threads 
        for i in eachindex(x)
            Δw, Δa, Δb = loss(rbm, J, PCD_state[i]; hparams, β, dev)
            if PCD
                PCD_state[i] = rbm.v |> cpu
            end

#             updateJ!(J, Δw, Δa, Δb; hparams)
            updateJAdam!(J, Δw, Δa, Δb, opt; hparams)

            enEpoch = enEpoch + en(rbm,J) |> cpu
            ΔwEpoch = ΔwEpoch + mean(Δw) |> cpu
            ΔaEpoch = ΔaEpoch + mean(Δa) |> cpu
            ΔbEpoch = ΔbEpoch + mean(Δb) |> cpu
        end
        append!(m.enList, enEpoch/size(x,1))
        append!(m.ΔwList, ΔwEpoch/size(x,1))
        append!(m.ΔaList, ΔaEpoch/size(x,1))
        append!(m.ΔbList, ΔbEpoch/size(x,1))
        if epoch % 1 == 0
            @info epoch, m.enList[end], m.ΔwList[end], m.ΔaList[end], m.ΔbList[end], β
            if plotSample
                genSample(rbm, J, hparams, m; num, t=t_samp, dev)
            end
        end
        if annealing
            β = β + 1/epochs
        end
        if PCD
            PCD_state = reshuffle(PCD_state; hparams)
        end
    end
    rbm, J, m, hparams
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