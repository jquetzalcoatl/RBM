using MLDatasets, Random

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
        train_x = cat(train_x_0, train_x_1, dims=3)
        idx = randperm(size(train_x,3))
        train_data = reshape(train_x, 28*28, :)[:,idx]
        x = [train_data[:,i] for i in Iterators.partition(1:size(train_data,2), hparams.batch_size)][1:end-1]
    end
    x
end