using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures
using CUDA
CUDA.device_reset!()
CUDA.device!(0)

include("../scripts/RAIS.jl")
include("../utils/train.jl")
include("../configs/yaml_loader.jl")
PATH = "/home/javier/Projects/RBM/NewResults/"
isdir(PATH) || mkpath(PATH)
config, _ = load_yaml_iter();
if config.phase_diagrams["gpu_bool"]
    dev = gpu
else
    dev = cpu
end



modelName = config.model_analysis["files"][1]
modelName = "CD-500-T1-BW-replica1"
modelName = "CD-FMNIST-500-T1000-BW-replica1-L"
modelName = "PCD-500-replica1"
modelName = "PCD-MNIST-500-lr-replica2"
rbm, J, m, hparams, opt = loadModel(modelName, dev, idx=100);
rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
dict = loadDict(modelName)
x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);


v,h = gibbs_sample(J, hparams, 100,1000)
lnum=10
mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
plot(heatmap(cpu(mat_rot)))

AIS(J, hparams, 500, 5000, 30, dev)