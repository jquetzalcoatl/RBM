begin
    using CUDA, Flux, HDF5
    using Base.Threads
    using StatsPlots
    CUDA.device_reset!()
    CUDA.device!(0)
    Threads.nthreads()
    ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"]="3GiB"
end

include("../utils/train.jl")

Random.seed!(1234);
d = Dict("bw"=>true, "tout" => 1.0)
rbm, J, m, hparams, opt = train(d, epochs=1000, nv=28*28, nh=400, batch_size=500, lr=0.0001, t=500, plotSample=true, 
   learnType="CD", gpu_usage = true, t_samp = 500, num=100, optType="Adam", numbers=collect(0:9), 
    savemodel=false, snapshot=10, annealing=true, β2=0.001)



########################
using JLD

d = JLD.load("/home/javier/Projects/RBM/Results/models/CD-500-T1000-5-BW-replica1-L/dict.jld")
d["dict"]
