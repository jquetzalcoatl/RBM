using Flux
include("utils/train.jl")

begin
    rbm, J, m, hparams = train(epochs=2)
    pEn = plot(m.enList, label="Energy")
    pLoss = plot(m.ΔwList, label="Loss w")
    pLoss = plot!(m.ΔaList, label="Loss a")
    pLoss = plot!(m.ΔbList, label="Loss b")
    plot(pEn, pLoss, layout=(2,1))
end

samp = reshape(Array{Float32}(sign.(rand(hparams.nv) .< σ.(J.w * rand(hparams.nh, 10) .+ J.a))), 28,28,:);
samp = reshape(Array{Float32}(sign.(rand(hparams.nv) .< σ.(J.w * rbm.h .+ J.a))), 28,28,:);

heatmap(reshape(samp[:,:,1:10], 28,28*10), size=(2500,300))

x = loadData(; hparams, dsName="MNIST01")

heatmap(reshape(reshape(x[1], 28, 28, :), 28,28*100), size=(2500,300))

heatmap(J.w)

plot(reshape(J.w,:), st=:hist, normalize=true)