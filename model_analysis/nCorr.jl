begin
using CUDA, Flux, HDF5
import JLD2
using Base.Threads
using StatsPlots
CUDA.device_reset!()
CUDA.device!(0)
Threads.nthreads()
end

begin
    include("../therm.jl")
    include("../configs/yaml_loader.jl")
    PATH = "/home/javier/Projects/RBM/Results/"
    dev = gpu
    β = 1.0
    config, _ = load_yaml_iter();
end


@doc raw"""
    N-body correlation function
    ar is an N x T array, where N are variables and T are measurements
    """
function n_body_correlation(ar, n)
    ar = ar .- mean(ar, dims=2)[:]
    if n == 2
        return cov(ar')
    elseif n==3
        N = size(ar,1)
        tmp = permutedims(cat([ar[i,:]' .* ar for i in 1:N]..., dims=3), (2,1,3))
        return cat([ar * tmp[:,:,i] for i in 1:N]..., dims=3)/(size(ar,2)-1)
    elseif n==4
        N = size(ar,1)
        tmp = permutedims(cat([ar[i,:]' .* ar for i in 1:N]..., dims=3), (2,1,3))
        return cat([cat([tmp[:,:,j]' * tmp[:,:,i] for i in 1:N]...,dims=3) for j in 1:N]..., dims=4)/(size(ar,2)-1)
    else
        return @warn "Something is odd..."
    end
end

function get_cov(z, order)
    z_s = z[:,1:10:10000]
    z_s = z_s .- mean(z_s, dims=2)[:]
    @time cov_tensor = n_body_correlation(z_s, order);
    cov_tensor
end

function _save_corr(cov2, cov3, cov4, modelname)
    FILE_PATH = PATH * config.model_analysis["output"] * modelname * "/"
    isdir(FILE_PATH) || mkpath(FILE_PATH)
    h5open(FILE_PATH * "correlation_tensors.h5", "w") do file
        write(file, "cov2", cov2) 
        write(file, "cov3", cov3)
        # write(file, "cov4", cov4)
    end
end

function _load_corr(PATH, FILENAME)
    path = PATH * FILENAME
    data = h5open(path);
    dataset = Dict()
    for key in keys(data)
        dataset[key] = h5read(path, key);
    end
    return dataset
end

function main()
    for model in config.model_analysis["files"]
        modelName = model
        bm, J, m, hparams, opt = loadModel(modelName, gpu);
        x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
        idx=100
        J = load("$(PATH)/models/$(modelName)/J/J_$(idx).jld", "J")
        J.w = gpu(J.w)
        J.b = gpu(J.b)
        J.a = gpu(J.a)
        F = LinearAlgebra.svd(J.w, full=true);
        v_val,h_val, x_val,y_val = data_val_samples(F, J, x_i, y_i; hparams) #(F, avg=false)
        z = size(y_val,1) <= size(x_val,1) ? cat(y_val, x_val, dims=1) : cat(x_val, y_val, dims=1)

        cov_mat = Dict()
        for i in 2:3
            cov_mat["$i"] = get_cov(z, i)
        end
        @info "Saving tensors"
        # _save_corr(cov_mat["2"], cov_mat["3"], cov_mat["4"], modelname)
        _save_corr(cov_mat["2"], cov_mat["3"], cov_mat["3"], modelName)
    end
end


main()



# #test
# z_s = reshape(Vector(1:30), 3,10) #z[1:10, 1:100]
# z_s = randn(4,1000000)
# begin
#     println("start")
#     n_body_correlation(z_s, 2)
# end
# ###########

# z_s = z[:,1:1000]
# z_s = z_s .- mean(z_s, dims=2)[:]
# @time cov2 = n_body_correlation(z_s, 2);
# heatmap(cov2)

# @time cov3 = n_body_correlation(z_s, 3);
# heatmap(cov3[:,:,2])

# reshape(cov3[1:end,1:end,1],:)
# plot(reshape(cov3[1:100,1:end,3],:))