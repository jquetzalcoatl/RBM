using ArgParse, CUDA, Flux
using Logging

include("./utils/train.jl")

function parseCommandLine()  
    s = ArgParseSettings(description = "Restricted Boltzmann Machine")
    @add_arg_table! s begin
      "--epochs", "-e"
        help = "Epochs"
        arg_type = Int64
        default = 2000
      "--msg", "-m"
        help = "Dir name"
        arg_type = String
        default = "0"
      "--bdir"
        help = "Base Dir name"
        arg_type = String
        default = "/home/javier/Projects/RBM/Results"
      "--dev"
        help = "Select device"
        arg_type = Int64
        default = 1
    end
  
    return parse_args(s) # the result is a Dict{String,Any
  end

function main()
    dictParse = parseCommandLine()
    modelname = dictParse["msg"]
    dict = loadDict(modelname, dictParse["bdir"]);
    path = dict["msg"]
    epochs=dictParse["epochs"]
    gpu_usage = dict["gpu"]
    
    if gpu_usage
        CUDA.device!(dictParse["dev"])
        ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"]="1GiB"
    end
    
    logger = SimpleLogger(open(dict["bdir"] * "/models/$path/log.txt", "a+"))
    global_logger(logger)
    
    rbm, J, m, hparams, opt = continuetrain(dict, modelname ; epochs, plotSample=false, t_samp=100, num=40)
    
    saveModel(rbm, J, m, hparams; opt, path, baseDir = dict["bdir"])
    dict["epochs"] = epochs
    saveDict(dict; path, baseDir = dict["bdir"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end