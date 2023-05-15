using MLDatasets, Random
using BSON: @save, @load

function loadData(; hparams, dsName="MNIST01")
    if dsName=="testing"
        #dummy DS
        dsSize=100
        train_data = rand([0,1],hparams.nv, dsSize)
        x = [train_data[:,i] for i in Iterators.partition(1:dsSize, hparams.batch_size)]
    elseif dsName=="MNIST01"    
        train_x = MLDatasets.MNIST(split=:train)[:].features
        train_y = MLDatasets.MNIST(split=:train)[:].targets
        train_x_0 = Array{Float32}(train_x[:, :, train_y .== 0] .≥ 0.5)
        train_x_1 = Array{Float32}(train_x[:, :, train_y .== 1] .≥ 0.5)
        train_x_5 = Array{Float32}(train_x[:, :, train_y .== 5] .≥ 0.5)
        train_x = cat(train_x_0, train_x_1, train_x_5, dims=3)
        @info size(train_x,3)
        idx = randperm(size(train_x,3))
        train_data = reshape(train_x, 28*28, :)[:,idx]
        x = [train_data[:,i] for i in Iterators.partition(1:size(train_data,2), hparams.batch_size)][1:end-1]
    end
    x
end

function saveModel(rbm, J, m, hparams; opt,  path = "0")
    isdir("./models/$path") || mkpath("./models/$path")
    @info "./models/$path"
    @save "./models/$path/RBM.bson" rbm
    @save "./models/$path/J.bson" J
    @save "./models/$path/m.bson" m
    @save "./models/$path/hparams.bson" hparams
    if hparams.optType == "Adam"
        @save "./models/$path/Opt.bson" opt
    end
end

function loadModel(path = "0")
    @load "./models/$path/RBM.bson" rbm
    @load "./models/$path/J.bson" J
    @load "./models/$path/m.bson" m
    @load "./models/$path/hparams.bson" hparams
    if hparams.optType == "Adam"
        @load "./models/$path/Opt.bson" opt
        return rbm, J, m, hparams, opt
    else
        return rbm, J, m, hparams, 0
    end
end