using ArgParse, CUDA, Flux
using Logging
CUDA.allowscalar(true)

include("./utils/train.jl")

function parseCommandLine()  
    s = ArgParseSettings(description = "Restricted Boltzmann Machine")
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
      "--plotsample", "-p"
        help = "Plot samples"
        arg_type = Bool
        default = false
      "--annealing", "-a"
        help = "Annealing?"
        arg_type = Bool
        default = true
      "--learn", "-D"
        help = "Type of learning? Rdm, CD, PCD"
        arg_type = String
        default = "Rdm"
      "--beta", "-T"
        help = "Inverse Temp"
        arg_type = Float64
        default = 0.0007
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
      "--opt"
        help = "Use ADAM? Or SGD"
        arg_type = String
        default = "Adam"
      "--numbers", "-n"
        help = "If using MNIST. Number labels to train on."
        nargs = '*'
        arg_type = Int64
        default = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      "--maxmem"
        help = "Specify max memory"
        arg_type = String
        default = "1GiB"
      "--bw"
        help = "Use Boltzmann weight after Gibbs sampling"
        arg_type = Bool
        default = true
      "--tout"
        help = "Final Temp if anneal = true"
        arg_type = Float64
        default = 10.0
      "--dataset"
        help = "Specify dataset"
        arg_type = String
        default = "MNIST01"
    end
  
    return parse_args(s) # the result is a Dict{String,Any
  end

function main()
    dict = parseCommandLine()
    epochs=dict["epochs"]
    nv=dict["nv"]
    nh=dict["nh"]
    batch_size=dict["batchsize"]
    lr=dict["lr"]
    t=dict["gibbs"]
    plotSample=false 
    annealing=dict["annealing"] 
    β2=dict["beta"]
    β = 1
    learnType=dict["learn"]
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

    saveDict(dict; path, baseDir = dict["bdir"])
    rbm, J, m, hparams, opt = train( dict ; epochs, nv, nh, batch_size, lr, t, plotSample, annealing, β2, learnType, gpu_usage, t_samp=t, num=100, optType=dict["opt"], snapshot=1, numbers, logging=true, io)
    # saveModel(rbm, J, m, hparams; opt, path, baseDir = dict["bdir"])
    saveDict(dict; path, baseDir = dict["bdir"])
    close(io)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # julia main.jl --bw true -a true -D CD -T 0.001 -m CD-500-T1000-BW-replica1-L --dev 1 --maxmem 3GiB -e 1000 &
    # julia main.jl --bw true -a true -D CD -T 0.001 --dataset FMNIST -m CD-FMNIST-500-T1000-BW-replica1-L --dev 1 --maxmem 3GiB -e 1000 &
    # julia main.jl --bw true -a true -D CD -T 0.001 -m CD-1200-500-T1000-BW-replica1 --dev 0 --maxmem 3GiB -e 1000 --tout 5.0 --nh 1200 &
    main()
end