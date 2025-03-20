using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures
using CUDA
CUDA.device_reset!()
CUDA.device!(0)

include("../utils/train.jl")
include("../scripts/RAIS.jl")
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

x_i

v,h = gibbs_sample(J, hparams, 100,1000)
lnum=10
mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
plot(heatmap(cpu(mat_rot)))


logZb_list, logZa_list = [],[]
for i in 1:10:100
    @info i
    rbm, J, m, hparams, opt = loadModel(modelName, dev, idx=i);
    logZb = AIS(J, hparams, 100, 500, 100.0, dev)
    logZa = RAIS(J, hparams, 100, 500, 100.0, dev)
    push!(logZa_list, logZa)
    push!(logZb_list, logZb)
end

plot(logZb_list)
plot!(logZa_list)

σ.(J.w' * v .+ J.b)

mean(v' * J.a + reshape(sum(log.(1 .+ exp.(J.w' * v .+ J.b)), dims=1),:))

J.w = cpu(J.w)
J.b = cpu(J.b)
J.a = cpu(J.a)
LL_numerator(x_i,J)

x_i

AIS(J, hparams, 100, 500, 100.0, dev)
ais = Array{Float64}[] #(undef, 0)
append!(ais, rand(10)[end])
ais[end] - ais[end]
ais[1]

using JLD2

ll = load("/home/javier/Projects/RBM/Results/Figs/PCD-500-replica1/partition_analytics.jld")
ll = load("/home/javier/Projects/RBM/Results/Figs/PCD-500-replica1/partition_cossio.jld")

plot(ll["lla"])
plot!(ll["llr"])

plot(ll["num"])

plot(- mean.(ll["rais"]))
plot!(-vcat(ll["rais"]...))

ll

using Integrals, QuadGK
f(x, p) = exp(cos(x)*p)
p = 5.4
domain = (0., 2π) # (lb, ub)
@time prob = IntegralProblem(f, domain, p)
@time sol = 2*solve(prob, QuadGKJL())

@time integral, error = quadgk(x -> f(x,p), 0, 2π)