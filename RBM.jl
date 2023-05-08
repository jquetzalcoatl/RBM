using Flux
include("utils/train.jl")


rbm, J, m, hparams = train( epochs=5, nv=28*28, nh=100, batch_size=100, lr=0.001, t=10, plotSample=true, annealing=true)

begin
    rbm, J, m, hparams = train(epochs=2, plotSample=true)
    pEn = plot(m.enList, label="Energy")
    pLoss = plot(m.ΔwList, label="Loss w")
    pLoss = plot!(m.ΔaList, label="Loss a")
    pLoss = plot!(m.ΔbList, label="Loss b")
    plot(pEn, pLoss, layout=(2,1))
end

sampSyn = reshape(Array{Float32}(sign.(rand(hparams.nv, 10) .< σ.(J.w * rand(hparams.nh, 10) .+ J.a))), 28,28,:);
sampH = reshape(Array{Float32}(sign.(rand(hparams.nv, 100) .< σ.(J.w * rbm.h .+ J.a))), 28,28,:);

heatmap(reshape(samp[:,:,1:10], 28,28*10), size=(2500,300))

x = loadData(; hparams, dsName="MNIST01")

heatmap(reshape(reshape(x[1], 28, 28, :), 28,28*100), size=(2500,300))

heatmap(J.w)

plot(reshape(J.w,:), st=:hist, normalize=true)