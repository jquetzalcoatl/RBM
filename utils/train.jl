using CUDA
using Plots, Statistics
using BSON: @load, @save

include("init.jl")
include("structs.jl")
include("loader.jl")
include("en.jl")

# Training function
function train( ; epochs=50, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=false)
    rbm, J, m, hparams = initModel(; nv, nh, batch_size, lr, t)
    x = loadData(; hparams, dsName="MNIST01")
    for epoch in 1:epochs
        enEpoch, ΔwEpoch, ΔaEpoch, ΔbEpoch = 0, 0, 0, 0
        Threads.@threads for i in eachindex(x)
            Δw, Δa, Δb = loss(rbm, J, x[i]; hparams)

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
            @info epoch, m.enList[end], m.ΔwList[end], m.ΔaList[end], m.ΔbList[end]
            if plotSample
                genSample(rbm, J, hparams, m; num = 4, t)
            end
        end
    end
    rbm, J, m, hparams
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end