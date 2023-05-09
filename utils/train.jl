using CUDA
using Plots, Statistics

include("init.jl")
include("structs.jl")
include("loader.jl")
include("en.jl")

# Training function
function train( ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false, annealing=false, β=1, PCD=true)
    rbm, J, m, hparams = initModel(; nv, nh, batch_size, lr, t)
    x = loadData(; hparams, dsName="MNIST01")
    PCD_state = x
    if annealing
        β = 0
    end 

    for epoch in 1:epochs
        enEpoch, ΔwEpoch, ΔaEpoch, ΔbEpoch = 0, 0, 0, 0
        
        Threads.@threads for i in eachindex(x)
            Δw, Δa, Δb = loss(rbm, J, PCD_state[i]; hparams, β)
            if PCD
                PCD_state[i] = rbm.v
            end

            updateJ!(J, Δw, Δa, Δb; hparams)

            enEpoch = enEpoch + en(rbm,J)
            ΔwEpoch = ΔwEpoch + mean(Δw)
            ΔaEpoch = ΔaEpoch + mean(Δa)
            ΔbEpoch = ΔbEpoch + mean(Δb)
        end
        append!(m.enList, enEpoch/size(x,1))
        append!(m.ΔwList, ΔwEpoch/size(x,1))
        append!(m.ΔaList, ΔaEpoch/size(x,1))
        append!(m.ΔbList, ΔbEpoch/size(x,1))
        if epoch % 1 == 0
            @info epoch, m.enList[end], m.ΔwList[end], m.ΔaList[end], m.ΔbList[end], β
            if plotSample
                genSample(rbm, J, hparams, m; num = 4, t)
            end
        end
        if annealing
            β = β + 1/epochs
        end
    end
    rbm, J, m, hparams
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end