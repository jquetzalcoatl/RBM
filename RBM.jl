using Flux
include("utils/train.jl")

nh=20, batch_size=50, lr=0.01, t=20
nh=20, batch_size=50, lr=0.05, t=20
nh=100, batch_size=50, lr=0.1, t=20
rbm, J, m, hparams = train( epochs=50, nv=28*28, nh=100, batch_size=50, lr=0.01, t=10, plotSample=true, annealing=false, β=1.0, PCD=true)

sampSyn = reshape(Array{Float32}(sign.(rand(hparams.nv, 10) .< σ.(J.w * rand(hparams.nh, 10) .+ J.a))), 28,28,:);
sampH = reshape(Array{Float32}(sign.(rand(hparams.nv, 100) .< σ.(J.w * rbm.h .+ J.a))), 28,28,:);

heatmap(reshape(samp[:,:,1:10], 28,28*10), size=(2500,300))

x = loadData(; hparams, dsName="MNIST01")

heatmap(reshape(reshape(x[1], 28, 28, :), 28,28*100), size=(2500,300))

heatmap(J.w)

plot(reshape(J.w,:), st=:hist, normalize=true)

saveModel(rbm, J, m, hparams; path = "2")

rbm, J, m, hparams = loadModel()

genSample(rbm, J, hparams, m; num = 100, t = 10, β = 0.8, mode = "test")