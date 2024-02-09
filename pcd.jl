import MLDatasets
using Statistics: mean, std, var
using Random: bitrand
# using ValueHistories: MVHistory, @trace
using RestrictedBoltzmannMachines: BinaryRBM, sample_from_inputs,
    initialize!, log_pseudolikelihood, pcd!, free_energy, sample_v_from_v
# using Plots

using CudaRBMs
# using CudaRBMs: cpu, gpu
# using CUDA

using ArgParse, CUDA #, Flux
using Logging
CUDA.allowscalar(true)

include("./utils/train.jl")

function parseCommandLine()  
    s = ArgParseSettings(description = "Restricted Boltzmann Machine trained via PCD")
    @add_arg_table! s begin
      "--nv", "-v"
        help = "Number of visible nodes"
        arg_type = Int64
        default = 28*28
      "--nh", "-i"
        help = "Number of hidden nodes"
        arg_type = Int64
        default = 500
      "--epochs", "-e"
        help = "Epochs"
        arg_type = Int64
        default = 100
      "--batchsize", "-b"
        help = "Batch Size"
        arg_type = Int64
        default = 500
      "--lr", "-l"
        help = "Learning rate"
        arg_type = Float64
        default = 0.0001
      "--gibbs", "-t"
        help = "Gibbs sampling length"
        arg_type = Int64
        default = 500
      "--msg", "-m"
        help = "Dir name"
        arg_type = String
        default = "0"
      "--bdir"
        help = "Base Dir name"
        arg_type = String
        default = "/home/javier/Projects/RBM/Results"
      "--gpu", "-g"
        help = "Use GPU?"
        arg_type = Bool
        default = true
      "--dev"
        help = "Select device"
        arg_type = Int64
        default = 1
      "--numbers", "-n"
        help = "If using MNIST. Number labels to train on."
        nargs = '*'
        arg_type = Int64
        default = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      "--maxmem"
        help = "Specify max memory"
        arg_type = String
        default = "3GiB"
    end
  
    return parse_args(s) # the result is a Dict{String,Any
  end

function main()
    dict = parseCommandLine()
    epochs=dict["epochs"]
    nv=dict["nv"]
    nh=dict["nh"]
    batchsize=dict["batchsize"]
    lr=dict["lr"]
    t=dict["gibbs"]
    path = dict["msg"]
    gpu_usage = dict["gpu"]
    numbers = dict["numbers"]
    
    isdir(dict["bdir"] * "/models/$path") || mkpath(dict["bdir"] * "/models/$path")
    io = open(dict["bdir"] * "/models/$path/log.txt", "w+")
    logger = SimpleLogger(io)
    global_logger(logger)
    
    if gpu_usage
        try
            CUDA.device!(dict["dev"])
        catch
            @warn "CUDA.device! prompt error. Skipping selecting device"
        end
        ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"]=dict["maxmem"]
    end
    
    if nv != 28*28
        @warn "Current script tailored for MNIST. Use default nv parameter value"
        exit()
    end

    saveDict(dict; path, baseDir = dict["bdir"])
    
    # Random.seed!(1234);
    hparams = HyperParams(nv=nv, nh=nh, batch_size=batchsize, gpu_usage=gpu_usage)
    
    x = loadData(; hparams, dsName="MNIST01", numbers, normalize=false, testset=false)
    train_x = CuArray(reshape(hcat(x...),28,28,:));
    
    rbm = CudaRBMs.gpu(BinaryRBM(Float32, (28,28), nh))
    initialize!(rbm, train_x)
    
    # batchsize = 500 #256
    iter_per_epoch = Int(floor(size(train_x,3)/batchsize))
    iters = iter_per_epoch * epochs #10000
    # history = MVHistory()
    rbmJ, J, m, hparams, rbmZJ = initModel(; nv, nh, batch_size=batchsize, lr, t, gpu_usage, optType="Adam")
    opt = initOptW(hparams, J);
    @time pcd!(
        rbm, train_x; steps=t, iters, batchsize,
        callback = function(; iter, _...)
            if iszero(iter % iter_per_epoch)
                # lpl = mean(log_pseudolikelihood(rbm, train_x))
                # @trace history iter lpl
                J.b = rbm.hidden.θ
                J.a = reshape(rbm.visible.θ,hparams.nv)
                J.w = reshape(rbm.w,hparams.nv,hparams.nh);
                saveModel(rbmJ, J, m, hparams; opt, path, epoch=Int(iter/iter_per_epoch))
            end
        end
    )
    saveDict(dict; path, baseDir = dict["bdir"])
    close(io)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end