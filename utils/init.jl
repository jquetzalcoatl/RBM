include("structs.jl")

function genRBM(args)
    rbm = RBM(rand(args.nv, args.batch_size), rand(args.nh, args.batch_size))
    rbm
end

function initWeights(args)
    W = randn(args.nv, args.nh)
    a = randn(args.nv)
    b = randn(args.nh)
    return Weights(W,a,b)
end

function initModelStats()
    mStats = ModelStats([],[],[],[])
    mStats
end

function initModel(; nv=10, nh=5, batch_size=4, lr=1.5)
    hparams = HyperParams(nv, nh, batch_size, lr)
    rbm = genRBM(hparams)
    J = initWeights(hparams)
    mStats = initModelStats()
    return rbm, J, mStats, hparams
end