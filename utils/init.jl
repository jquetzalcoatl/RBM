using CUDA, Flux
include("structs.jl")
include("adamOpt.jl")

function selectDev(args)
    if args.gpu_usage
        dev = gpu
    else
        dev = cpu
    end
    dev
end

function genRBM(args)
    dev = selectDev(args)
    rbm = RBM(rand(args.nv, args.batch_size) |> dev, rand(args.nh, args.batch_size) |> dev)
    rbm
end

function initWeights(args)
    dev = selectDev(args)
    W = randn(args.nv, args.nh) .* 0.1 / √(args.nh) |> dev
    a = randn(args.nv) .* 0.1 / √args.nv |> dev
    b = randn(args.nh) .* 0.1 / √args.nh |> dev
    return Weights(W,a,b)
end

function initModelStats()
    mStats = ModelStats([[] for i in 1:length(fieldnames(ModelStats))]...)
    mStats
end

function initModel(; nv=10, nh=5, batch_size=4, lr=1.5, γ=0.001, t=10, gpu_usage = false, optType="SGD")
    hparams = HyperParams(nv, nh, batch_size, lr, γ, t, gpu_usage, optType)
    rbm = genRBM(hparams)
    J = initWeights(hparams)
    mStats = initModelStats()
    rbmZ = genRBM(hparams)
    return rbm, J, mStats, hparams, rbmZ
end

function initOptW(args, J)
    dev = selectDev(args)
    optW = Adam(J.w, args.lr; dev)
    optA = Adam(J.a, args.lr; dev)
    optB = Adam(J.b, args.lr; dev)
    opt = WeightOpt(optW, optA, optB)
    return opt
end