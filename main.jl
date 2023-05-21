using ArgParse, CUDA, Flux

# include("./utils/init.jl")
# include("./utils/structs.jl")
# include("./utils/loader.jl")
# include("./utils/en.jl")
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
        default = 2000
      "--batchsize", "-b"
        help = "Batch Size"
        arg_type = Int64
        default = 500
      "--lr", "-l"
        help = "Learning rate"
        arg_type = Float64
        default = 0.001
      "--gibbs", "-t"
        help = "Gibbs sampling length"
        arg_type = Int64
        default = 10
      "--plotsample", "-p"
        help = "Plot samples"
        arg_type = Bool
        default = false
      "--annealing", "-a"
        help = "Annealing?"
        arg_type = Bool
        default = false
      "--pcd", "-D"
        help = "PCD?"
        arg_type = Bool
        default = true
      "--beta", "-c"
        help = "Inverse Temp"
        arg_type = Float64
        default = 1.0
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
        default = false
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
        default = [0, 1]
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
    β=dict["beta"]
    PCD=dict["pcd"]
    path = dict["msg"]
    gpu_usage = dict["gpu"]
    numbers = dict["numbers"]
    if gpu_usage
        CUDA.device!(dict["dev"])
    end
    rbm, J, m, hparams, opt = train( ; epochs, nv, nh, batch_size, lr, t, plotSample, annealing, β, PCD, gpu_usage, t_samp=100, num=40, optType=dict["opt"], numbers)
#     if dict["opt"]
#         rbm, J, m, hparams, opt = trainAdam( ; epochs, nv, nh, batch_size, lr, t, plotSample, annealing, β, PCD, gpu_usage, t_samp=100, num=40, optType="Adam")
#     else
#         rbm, J, m, hparams, opt = train( ; epochs, nv, nh, batch_size, lr, t, plotSample, annealing, β, PCD, gpu_usage, t_samp=100, num=40, optType="SGD")
#     end
    saveModel(rbm, J, m, hparams; opt, path, basedir = dict["bdir"])
    loadModel(path, gpu)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end