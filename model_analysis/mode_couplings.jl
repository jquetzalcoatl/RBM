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
    modelName = config.model_analysis["files"][1]
    rbm, J, m, hparams, opt = loadModel(modelName, gpu);
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    # idx=100
    # J = load("$(PATH)/models/$(modelName)/J/J_$(idx).jld", "J")
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
    v_synth,h_synth = gibbs_sampling(v_synth,h_synth,J; mcs=500, dev0=gpu)
    x_synth = cpu(F.U' * v_synth)
    y_synth = cpu(F.Vt * h_synth);
    z_synth = size(y_synth,1) <= size(x_synth,1) ? cat(y_synth, x_synth, dims=1) : cat(x_synth, y_synth, dims=1)
    lnum=15
    mat = cat([cat([reshape(v_synth[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
    mat_rot = reverse(transpose(mat), dims=1)
    heatmap(cpu(mat_rot), size=(900,900))
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

#Kurtosis
using StatsBase
len = 900
plot([kurtosis(z_val[i,:]) for i in 1:len], st=:histogram)
plot!([kurtosis(randn(10000)) for i in 1:len], st=:histogram)

#RW
function rw(v,h,J; mcs=5000, dev0=cpu, evry=10)
    β=1
    dev = gpu
    nh = size(h,1)
    nv = size(v,1)
    num= size(v,2)
    # v = gpu(v[:,num])
    # h = gpu(h[:,num])
    v = gpu(v)
    h = gpu(h)
    J.w = gpu(J.w)
    J.b = gpu(J.b)
    J.a = gpu(J.a)
    F = LinearAlgebra.svd(J.w, full=true);

    x = zeros(size(v,1),size(v,2),Int(floor(mcs/evry))+1)
    y = zeros(size(h,1),size(h,2),Int(floor(mcs/evry))+1)
    @info size(y)
    
    counter = 1
    for i in 1:mcs
        h = Array{Float32}(sign.(rand(nh, num) |> dev .< σ.(β .* (J.w' * v .+ J.b)))) |> dev
        
        v = Array{Float32}(sign.(rand(nv, num) |> dev .< σ.(β .* (J.w * h .+ J.a)))) |> dev 
        if i % evry == 0
            y[:,:,counter+1] = cpu(F.Vt * h)
            x[:,:,counter+1] = cpu(F.U' * v)
            counter = counter + 1
        end
    end
    return dev0(x),dev0(y), dev0(v),dev0(h)
end

# dims are variable x samples x time
JJ = initWeights(hparams)

begin
    mc_steps=10000
    save_every = 10
    ar_size = Int(mc_steps / save_every) + 1
    sample_size = 400
    x,y, v,h = rw(zeros(hparams.nv,sample_size),zeros(hparams.nh,sample_size),J; mcs=mc_steps, dev0=cpu, evry=save_every);
    # x,y, v,h = rw(rand([0,1],hparams.nv,sample_size),rand([0,1],hparams.nh,sample_size),J; mcs=mc_steps, dev0=cpu, evry=save_every);



    lnum=Int(sqrt(sample_size))
    mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
    mat_rot = reverse(transpose(mat), dims=1)
    f1 = heatmap(cpu(mat_rot), size=(900,900))
end

begin
    f2 = plot(reshape(mean(x,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
        xlabel="MC step (x $(save_every))", ylabel="⟨x⟩", frame=:box, title=modelName, opacity=0.5);

    f3 = plot(reshape(std(x,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
        xlabel="MC step (x $(save_every))", ylabel="std(x)", frame=true, opacity=0.5);

    f4 = plot(reshape(mean(y,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none, 
        xlabel="MC step (x $(save_every))", ylabel="⟨y⟩", frame=:box, opacity=0.5);

    f5 = plot(reshape(std(y,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
        xlabel="MC step (x $(save_every))", ylabel="std(y)", frame=:box, opacity=0.5);

    f6 = plot([kurtosis(x[i,:,j]) for i in 1:hparams.nv, j in 2:ar_size]', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
    xlabel="MC step (x $(save_every))", ylabel="kutorsis(x)", frame=:box, opacity=0.5);

    f7 = plot([kurtosis(y[i,:,j]) for i in 1:hparams.nh, j in 2:ar_size]', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
    xlabel="MC step (x $(save_every))", ylabel="kutorsis(y)", frame=:box, opacity=0.5);

    plot(f1,f2,f3, f6,f4,f5, f7, size=(1200,900))
end

[kurtosis(x[i,:,j]) for i in 1:hparams.nv, j in 2:ar_size]

hcat([diag(x[i,:,:]' * y[i,:,:] / sample_size) for i in 1:500]...)
f2 = plot(hcat([diag(x[i,:,:]' * y[i,:,:] / sample_size) for i in 1:500]...), color=[RGB(1 - i/hparams.nv, 0, i/hparams.nv) for i in 1:hparams.nv]', legend=:none,
        xlabel="MC step (x $(save_every))", ylabel="⟨x⟩", frame=:box, title=modelName, opacity=0.5)


#################
begin
    plt_list = []
    for i in 1:10
        push!(plt_list, plot(x[i,:,:]', legend=:none, s=:auto, lw=1.5, frame=:box, label=i))
    end

    plot(plt_list..., size=(900,900))
end

begin
    plt_list = []
    for i in 1:10
        push!(plt_list, plot(y[i,:,:]', legend=:none, s=:auto, lw=1.5, frame=:box, label=i))
    end

    plot(plt_list..., size=(900,900))
end
