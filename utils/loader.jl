using MLDatasets, Random, JLD
# using BSON: @save, @load
include("init.jl")
include("adamOpt.jl")
include("structs.jl")

function loadData(; hparams, dsName="MNIST01", numbers = [0,1])
    if dsName=="testing"
        #dummy DS
        dsSize=100
        train_data = rand([0,1],hparams.nv, dsSize)
        x = [train_data[:,i] for i in Iterators.partition(1:dsSize, hparams.batch_size)]
    elseif dsName=="MNIST01"    
        train_x = MLDatasets.MNIST(split=:train)[:].features
        train_y = MLDatasets.MNIST(split=:train)[:].targets

        train_x_samp = Array{Float32}(train_x[:, :, train_y .== numbers[1]] .≥ 0.5)
        if size(numbers,1)>1
            for idx in numbers[2:end]
                train_x_tmp = Array{Float32}(train_x[:, :, train_y .== idx] .≥ 0.5)
                train_x_samp = cat(train_x_samp, train_x_tmp, dims=3)
            end
        end
        train_x = train_x_samp
        # train_x_0 = Array{Float32}(train_x[:, :, train_y .== 0] .≥ 0.5)
        # train_x_1 = Array{Float32}(train_x[:, :, train_y .== 1] .≥ 0.5)
        # train_x_5 = Array{Float32}(train_x[:, :, train_y .== 5] .≥ 0.5)
        # train_x = cat(train_x_0, train_x_1, train_x_5, dims=3)
        # train_x = cat(train_x_0, dims=3)
        @info size(train_x,3)
        idx = randperm(size(train_x,3))
        train_data = reshape(train_x, 28*28, :)[:,idx]
        x = [train_data[:,i] for i in Iterators.partition(1:size(train_data,2), hparams.batch_size)][1:end-1]
    end
    x
end

function saveModel(rbm, J, m, hparams; opt,  path = "0", baseDir = "/home/javier/Projects/RBM/Results")
    isdir(baseDir * "/models/$path") || mkpath(baseDir * "/models/$path")
    # @info "$(baseDir)/models/$path"
    save("$(baseDir)/models/$path/RBM.jld", "rbm", RBM(rbm.v |> cpu, rbm.h |> cpu) )
    save("$(baseDir)/models/$path/J.jld", "J", Weights(J.w |> cpu, J.a |> cpu, J.b |> cpu) )
    save("$(baseDir)/models/$path/m.jld", "m", m)
    save("$(baseDir)/models/$path/hparams.jld", "hparams", hparams)
    if hparams.optType == "Adam"
        optW = Adam(opt.w.theta |> cpu, opt.w.m |> cpu, opt.w.v |> cpu, opt.w.b1, opt.w.b2, opt.w.a, opt.w.eps, opt.w.t)
        opta = Adam(opt.a.theta |> cpu, opt.a.m |> cpu, opt.a.v |> cpu, opt.a.b1, opt.a.b2, opt.a.a, opt.a.eps, opt.a.t)
        optb = Adam(opt.b.theta |> cpu, opt.b.m |> cpu, opt.b.v |> cpu, opt.b.b1, opt.b.b2, opt.b.a, opt.b.eps, opt.b.t)
        
        # optO.w.theta = optO.w.theta |> cpu
        # optO.w.m = optO.w.m |> cpu
        # optO.w.v = optO.w.v |> cpu
        # optO.a.theta = optO.a.theta |> cpu
        # optO.a.m = optO.a.m |> cpu
        # optO.a.v = optO.a.v |> cpu
        # optO.b.theta = optO.b.theta |> cpu
        # optO.b.m = optO.b.m |> cpu
        # optO.b.v = optO.b.v |> cpu
        save("$(baseDir)/models/$path/Opt.jld", "opt", WeightOpt(optW, opta, optb))
    end
end

function loadModel(path = "0", dev = cpu, baseDir = "/home/javier/Projects/RBM/Results")
    isdir(baseDir * "/models/$path") || mkpath(baseDir * "/models/$path")
    @info "$(baseDir)/models/$path"
    rbm = load("$(baseDir)/models/$path/RBM.jld", "rbm")
    rbm = RBM([getfield(rbm, field) |> dev for field in fieldnames(RBM)]...)
    J = load("$(baseDir)/models/$path/J.jld", "J")
    J = Weights([getfield(J, field) |> dev for field in fieldnames(Weights)]...)
    m = load("$(baseDir)/models/$path/m.jld", "m")
    hparams = load("$(baseDir)/models/$path/hparams.jld", "hparams")
    if hparams.optType == "Adam"
        opt = load("$(baseDir)/models/$path/Opt.jld", "opt")
        opt.w.theta = opt.w.theta |> dev
        opt.w.m = opt.w.m |> dev
        opt.w.v = opt.w.v |> dev
        opt.a.theta = opt.a.theta |> dev
        opt.a.m = opt.a.m |> dev
        opt.a.v = opt.a.v |> dev
        opt.b.theta = opt.b.theta |> dev
        opt.b.m = opt.b.m |> dev
        opt.b.v = opt.b.v |> dev
        return rbm, J, m, hparams, opt
    else
        return rbm, J, m, hparams, 0
    end
end

function saveDict(dict; path = "0", baseDir = "/home/javier/Projects/RBM/Results")
    isdir(baseDir * "/models/$path") || mkpath(baseDir * "/models/$path")
    save("$(baseDir)/models/$path/dict.jld", "dict", dict)
end

function loadDict(path = "0", baseDir = "/home/javier/Projects/RBM/Results")
    dict = load("$(baseDir)/models/$path/dict.jld", "dict")
    return dict
end
    
    

# function saveModel(rbm, J, m, hparams; opt,  path = "0", baseDir = "/home/javier/Projects/RBM/Results")
#     isdir(baseDir * "/models/$path") || mkpath(baseDir * "/models/$path")
#     @info "$(baseDir)/models/$path"
#     @save "$(baseDir)/models/$path/RBM.bson" rbm
#     @save "$(baseDir)/models/$path/J.bson" J
#     @save "$(baseDir)/models/$path/m.bson" m
#     @save "$(baseDir)/models/$path/hparams.bson" hparams
#     if hparams.optType == "Adam"
#         @save "$(baseDir)/models/$path/Opt.bson" opt
#     end
# end

# function loadModel(path = "0", dev = cpu, baseDir = "/home/javier/Projects/RBM/Results")
#     @load "$(baseDir)/models/$path/RBM.bson" rbm
#     rbm = RBM([getfield(rbm, field) |> dev for field in fieldnames(RBM)]...)
#     @load "$(baseDir)/models/$path/J.bson" J
#     J = Weights([getfield(J, field) |> dev for field in fieldnames(Weights)]...)
#     @load "$(baseDir)/models/$path/m.bson" m
#     @load "$(baseDir)/models/$path/hparams.bson" hparams
#     if hparams.optType == "Adam"
#         @load "$(baseDir)/models/$path/Opt.bson" opt
#         opt.w.m = opt.w.m |> dev
#         opt.w.v = opt.w.v |> dev
#         opt.a.m = opt.a.m |> dev
#         opt.a.v = opt.a.v |> dev
#         opt.b.m = opt.b.m |> dev
#         opt.b.v = opt.b.v |> dev
# #         opt = Adam([getfield(opt, field) |> dev for field in fieldnames(Adam)]...)
#         return rbm, J, m, hparams, opt
#     else
#         return rbm, J, m, hparams, 0
#     end
# end