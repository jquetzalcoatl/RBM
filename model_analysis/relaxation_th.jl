begin
    using CUDA, Flux, HDF5
    using Base.Threads
    using StatsPlots
    using BSplineKit
    using QuadGK
    using FFTW, CurveFit
    CUDA.device_reset!()
    CUDA.device!(1)
    Threads.nthreads()
end

include("../utils/train.jl")

Random.seed!(1234);
rbm, J, m, hparams, rbmZ = initModel(nv=5, nh=5, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")

hparams.nh
hparams.nv

using LinearAlgebra

LinearAlgebra.SVD