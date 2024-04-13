begin
    using CUDA, Flux, HDF5
    using Base.Threads
    using StatsPlots
    CUDA.device_reset!()
    CUDA.device!(0)
    Threads.nthreads()
end

include("../utils/train.jl")

Random.seed!(1234);
d = Dict("bw"=>false)
rbm, J, m, hparams, opt = train(d, epochs=50, nv=28*28, nh=500, batch_size=500, lr=0.0001, t=100, plotSample=true, 
    annealing=false, learnType="CD", β=1, β2 = 1, gpu_usage = false, t_samp = 100, num=100, optType="Adam", numbers=[1,5], 
    savemodel=false, snapshot=1)

rbm, J, m, hparams, rbmZ = initModel(nv=10, nh=5, batch_size=500, lr=1.5, t=10, gpu_usage = false, optType="Adam")
# opt = initOptW(hparams, J);


function gibbs_sampling(v,h,J; mcs=5000, dev0=cpu)
    β=1
    dev = gpu
    nh = size(h,1)
    nv = size(v,1)
    num= minimum([size(v,2), 1000])
    v = gpu(v[:,num])
    h = gpu(h[:,num])
    J.w = gpu(J.w)
    J.b = gpu(J.b)
    J.a = gpu(J.a)
    
    for i in 1:mcs
        h = Array{Float32}(sign.(rand(nh, num) |> dev .< σ.(β .* (J.w' * v .+ J.b)))) |> dev
        v = Array{Float32}(sign.(rand(nv, num) |> dev .< σ.(β .* (J.w * h .+ J.a)))) |> dev 
    end
    return dev0(v),dev0(h)
end

function AIS()
    lnZa = log(1 + exp)
end

gibbs_sampling(rbm.v, rbm.h, J)

