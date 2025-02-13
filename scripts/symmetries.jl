using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures
using CUDA
CUDA.device_reset!()
CUDA.device!(0)

include("../utils/init.jl")
include("../scripts/PhaseAnalysis.jl")

PATH = "/home/javier/Projects/RBM/NewResults/"

modelName = config.model_analysis["files"][6]
modelName = "CD-500-T1-BW-replica1"
modelName = "CD-FMNIST-500-T1000-BW-replica1-L"
modelName = "PCD-100-replica1"
modelName = "PCD-MNIST-500-lr-replica2"
rbm, J, m, hparams, opt = loadModel(modelName, dev, idx=100);
dict = loadDict(modelName)

rbm, J, m, hparams, rbmZ = initModel(nv=728, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")

std(J.w)


W = randn(hparams.nv,hparams.nh) .* 0.01
for i in 1:100
    W = W .+ randn(hparams.nv,hparams.nh) .* 0.01
end
W = W/100.

mean((cpu(J.w) .- W) .^ 2)