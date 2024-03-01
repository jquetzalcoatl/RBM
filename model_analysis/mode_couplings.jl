begin
    using CUDA, Flux, HDF5
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

begin
    modelName = config.model_analysis["files"][10]
    rbm, J, m, hparams, opt = loadModel(modelName, gpu);
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    idx=100
    J = load("$(PATH)/models/$(modelName)/J/J_$(idx).jld", "J")
    J.w = gpu(J.w)
    J.b = gpu(J.b)
    J.a = gpu(J.a)
    F = LinearAlgebra.svd(J.w, full=true);
    v_val,h_val, x_val,y_val = data_val_samples(F, J, x_i, y_i; hparams) #(F, avg=false)
    z_val = size(y_val,1) <= size(x_val,1) ? cat(y_val, x_val, dims=1) : cat(x_val, y_val, dims=1)
end

###Synthetic data
begin
    v_synth = sign.(rand(hparams.nv, 500) .< 0.5) |> dev ;
    h_synth = sign.(rand(hparams.nh, 500) .< 0.5) |> dev ;
    v_synth,h_synth = gibbs_sampling(v_synth,h_synth,J; mcs=5000, dev0=gpu)
    x_synth = cpu(F.U' * v_synth)
    y_synth = cpu(F.Vt * h_synth);
    z_synth = size(y_synth,1) <= size(x_synth,1) ? cat(y_synth, x_synth, dims=1) : cat(x_synth, y_synth, dims=1)
    lnum=15
    mat = cat([cat([reshape(v_synth[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
    mat_rot = reverse(transpose(mat), dims=1)
    heatmap(mat_rot, size=(900,900))
end

begin
    plt_list = []
    for i in 1:8
        push!(plt_list, plot(z_val[1,:], z_val[hparams.nh + i,:], s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.1, frame=:box))
    end

    for i in 1:8
        push!(plt_list, plot(z_val[1,:], z_val[hparams.nh + i,:], c=:orange, s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.1, frame=:box))
    end
    plot(plt_list..., size=(700,500))
end

corr = cov(z_val');
heatmap(corr, size=(900,900))
plot(abs.(corr[501,:]))
d,vecs = eigen(corr)

plot(abs.(d[200:end]), yscale=:log10, legend=false, s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.1, frame=:box)

plot(abs.(vecs[:,end]) ./ maximum(abs.(vecs[:,end])), size=(900,900))
plot!(abs.(vecs[:,end-1]) ./ maximum(abs.(vecs[:,end-1])), size=(900,900))
most_coupled_modes = sortperm(abs.(vecs[:,end]), rev=true)[1:12]

most_coupled_modes = sortperm(abs.(vecs[:,end-1]), rev=true)[1:12]

begin
    plt_list = []
    for i in most_coupled_modes
        push!(plt_list, plot(z_val[501,:], z_val[i,:], s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.1, frame=:box, label=i))
    end

    for i in most_coupled_modes
        push!(plt_list, plot(z_val[1,:], z_val[i,:], s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.1, frame=:box, label=i, c=:orange))
    end
    plot(plt_list..., size=(900,900))
end