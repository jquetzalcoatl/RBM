using ArgParse

# include("./utils/init.jl")
# include("./utils/structs.jl")
# include("./utils/loader.jl")
# include("./utils/en.jl")
include("./utils/train.jl")

function parseCommandLine()  
    s = ArgParseSettings(description = "First Passage Process w/ Stochastic Resetting Via An External Potential")
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
        default = 500
      "--batchsize", "-b"
        help = "Batch Size"
        arg_type = Int64
        default = 100
      "--lr", "-l"
        help = "Learning rate"
        arg_type = Float64
        default = 0.01
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
        default = true
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
    rbm, J, m, hparams = train( ; epochs, nv, nh, batch_size, lr, t, plotSample, annealing, β, PCD)
    saveModel(rbm, J, m, hparams; path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end