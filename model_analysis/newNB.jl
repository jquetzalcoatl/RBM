begin
    using CUDA, Flux, HDF5
    using Base.Threads
    using StatsPlots
    CUDA.device_reset!()
    CUDA.device!(0)
    Threads.nthreads()
end
#Does not work
import TensorCrossInterpolation as TCI
include("../utils/train.jl")

include("../scripts/exact_partition.jl")



Random.seed!(1234);
# d = Dict("bw"=>false)
# rbm, J, m, hparams, opt = train(d, epochs=50, nv=28*28, nh=500, batch_size=500, lr=0.0001, t=100, plotSample=true, 
    # annealing=false, learnType="CD", β=1, β2 = 1, gpu_usage = false, t_samp = 100, num=100, optType="Adam", numbers=[1,5], 
    # savemodel=false, snapshot=1)

function partition_function_tci(J)
    W = hcat(vcat(J.w, zeros(size(J.w))), vcat(zeros(size(J.w)),J.w)) |> cpu
    self_fields = vcat(J.a,J.b) |> cpu
    f(v) = exp( ((v .- 1)' * self_fields + (v .- 1)' * W * (v .- 1)))
    localdims = fill(2, size(J.a)[1]*2)    # There are 5 tensor indices, each with values 1...10
    tolerance = 1e-15
    @time tci, ranks, errors = TCI.crossinterpolate2(Float64, f, localdims; tolerance=tolerance)
    return tci, ranks, errors
end

# Initialize random RBM. self-fields and couplings are sampled from Gaussian distribution
rbm, J, m, hparams, rbmZ = initModel(nv=5, nh=5, batch_size=500, lr=1.5, t=10, gpu_usage = false, optType="Adam")

# This function computes the exact partition function for small RBM
partition_function(J)

# TCI implementation from Julia documentation
tci, ranks, errors = partition_function_tci(J)
sum(tci)

# Lets check for a simple model where W=0 and self-fields=2.1
# Z = (1+ exp(2.1))^10
(1+ exp(2.1))^10
J.w = zeros(5,5)
J.a .= 2.1
J.b .= 2.1
partition_function(J)
tci, ranks, errors = partition_function_tci(J)
sum(tci)


# We now test TCI for when the weights are uniformly distributed, which is more challenging, 
# since the model is near a spin-glass-Ferro Phase Transition
J.w = rand(5,5) .* 0.1
partition_function(J)
tci, ranks, errors = partition_function_tci(J)
sum(tci)