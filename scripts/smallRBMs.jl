begin
    using CUDA, Flux, HDF5
    using Base.Threads
    using StatsPlots
    CUDA.device_reset!()
    CUDA.device!(0)
    Threads.nthreads()
end

include("../utils/train.jl")

include("../scripts/exact_partition.jl")
include("../scripts/gaussian_partition.jl")
include("../scripts/gaussian_orth_partition.jl")
include("../scripts/RAIS.jl")

Random.seed!(1234);
# d = Dict("bw"=>false)
# rbm, J, m, hparams, opt = train(d, epochs=50, nv=28*28, nh=500, batch_size=500, lr=0.0001, t=100, plotSample=true, 
    # annealing=false, learnType="CD", β=1, β2 = 1, gpu_usage = false, t_samp = 100, num=100, optType="Adam", numbers=[1,5], 
    # savemodel=false, snapshot=1)

rbm, J, m, hparams, rbmZ = initModel(nv=5, nh=5, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")

function smallRBM(replicas=50, max_size=5, samples=50, mcs=500, nbeta=20)
    # replicas = 50
    part_func = zeros(max_size-2+1,replicas+1)
    part_func_G = zeros(max_size-2+1,replicas+1)
    part_func_AIS = zeros(max_size-2+1,2)
    part_func_RAIS = zeros(max_size-2+1,2)
    for (i,s) in enumerate(2:max_size)
        @info "RBM size $s"
        zs = []
        zs_G = []
        rbm, J, m, hparams, rbmZ = initModel(nv=s, nh=s, batch_size=1, lr=1.5, t=10, gpu_usage = true, optType="Adam")
        for r in 1:replicas
            rbm, J, m, hparams, rbmZ = initModel(nv=s, nh=s, batch_size=1, lr=1.5, t=10, gpu_usage = true, optType="Adam")

            push!(zs,partition_function(J))

            push!(zs_G, pf_Gauss_beta(J, hparams))
        end

        part_func[i,:] = hcat(s,zs') #vcat(part_func, hcat(s,zs'))
        part_func_G[i,:] = hcat(s,zs_G') #vcat(part_func_G, hcat(s,zs_G'))

        part_func_AIS[i,:] = hcat(s,AIS(J, hparams, samples, mcs, nbeta)) #vcat(part_func_AIS, hcat(s,AIS(J, hparams, samples, mcs, nbeta)))
        part_func_RAIS[i,:] = hcat(s,RAIS(J, hparams, samples, mcs, nbeta)) #vcat(part_func_RAIS, hcat(s,RAIS(J, hparams, samples, mcs, nbeta)))
    end
    return part_func, part_func_G, part_func_AIS, part_func_RAIS
end


part_func, part_func_G, part_func_AIS, part_func_RAIS = smallRBM(50, 10, 50, 500, 20)
begin
    # F = -kTln(Z)
    # F = E - TS
    # ln(Z) - S = - F/kT - S = - E/kT 
    fig = plot(part_func[:,1], log.(mean(part_func[:,2:end], dims=2)[:]) .- 2 .* part_func[:,1] .* log(2), 
        yerr=log.(std(part_func[:,2:end], dims=2)[:]), frame=:box,
        label="Exact", s=:auto, markershapes = :circle, lw=0.5, markerstrokewidth=0.1, ms=15)

    fig = plot!(part_func_G[:,1], log.(mean(part_func_G[:,2:end], dims=2)[:]) .- 2 .* part_func[:,1] .* log(2), 
        yerr=log.(std(part_func_G[:,2:end], dims=2)[:]), color=:red, frame=:box,
        label="Approximation", s=:auto, markershapes = :star5, lw=0.5, markerstrokewidth=0.1, ms=10)

    fig = plot!(part_func_AIS[:,1], part_func_AIS[:,2] .- 2 .* part_func[:,1] .* log(2), 
        label="AIS", s=:auto, markershapes = :diamond, lw=0.5, markerstrokewidth=0.1, ms=10)

    fig = plot!(part_func_RAIS[:,1], part_func_RAIS[:,2] .- 2 .* part_func[:,1] .* log(2),
        label="RAIS", s=:auto, markershapes = :hexagon, lw=0.5, markerstrokewidth=0.1, ms=5)

    # plot!(part_func[:,1], 2 .* part_func[:,1] .* log(2))
    fig = plot!(xlabel="Number of nodes p/ partiton", ylabel="ln(Z) - Entropy", legend=:topleft, size=(700,500))
    savefig(fig, "/home/javier/Projects/RBM/Results/smallRBMs.png")
end

function not_so_smallRBM(replicas=50, min_size=20, max_size=30, samples=50, mcs=500, nbeta=20)
   part_func_G = zeros(max_size-min_size+1,replicas+1)
   part_func_AIS = zeros(max_size-min_size+1,2)
   part_func_RAIS = zeros(max_size-min_size+1,2)
   for (i,s) in enumerate(min_size:max_size)
       @info "RBM size $s"
    #    zs = []
       zs_G = []
       rbm, J, m, hparams, rbmZ = initModel(nv=s, nh=s, batch_size=1, lr=1.5, t=10, gpu_usage = true, optType="Adam")
       for r in 1:replicas
           rbm, J, m, hparams, rbmZ = initModel(nv=s, nh=s, batch_size=1, lr=1.5, t=10, gpu_usage = true, optType="Adam")

           push!(zs_G, log_pf_Gauss_beta(J, hparams))
       end
       part_func_G[i,:] = hcat(s,zs_G') #vcat(part_func_G, hcat(s,zs_G'))

       part_func_AIS[i,:] = hcat(s,AIS(J, hparams, samples, mcs, nbeta)) #vcat(part_func_AIS, hcat(s,AIS(J, hparams, samples, mcs, nbeta)))
       part_func_RAIS[i,:] = hcat(s,RAIS(J, hparams, samples, mcs, nbeta)) #vcat(part_func_RAIS, hcat(s,RAIS(J, hparams, samples, mcs, nbeta)))
   end
   return part_func_G, part_func_AIS, part_func_RAIS
end

part_func_G
part_func_AIS
part_func_G, part_func_AIS, part_func_RAIS = not_so_smallRBM(50, 200, 210, 50, 500, 20)
begin
    # F = -kTln(Z)
    # F = E - TS
    # ln(Z) - S = - F/kT - S = - E/kT 
    # plot(part_func[:,1], log.(mean(part_func[:,2:end], dims=2)[:]) .- 2 .* part_func[:,1] .* log(2), 
        # yerr=log.(std(part_func[:,2:end], dims=2)[:]), frame=:box,
        # label="Exact", s=:auto, markershapes = :circle, lw=0.5, markerstrokewidth=0.1, ms=10)

    fig = plot(part_func_G[:,1], mean(part_func_G[:,2:end], dims=2)[:] .- 2 .* part_func_G[:,1] .* log(2), 
        yerr=log.(std(part_func_G[:,2:end], dims=2)[:]), color=:red, frame=:box,
        label="Approximation", s=:auto, markershapes = :star5, lw=0.5, markerstrokewidth=0.1, ms=10)

    fig = plot!(part_func_AIS[:,1], part_func_AIS[:,2] .- 2 .* part_func_G[:,1] .* log(2), 
        label="AIS", s=:auto, markershapes = :diamond, lw=0.5, markerstrokewidth=0.1, ms=10)

    fig = plot!(part_func_RAIS[:,1], part_func_RAIS[:,2] .- 2 .* part_func_G[:,1] .* log(2),
        label="RAIS", s=:auto, markershapes = :hexagon, lw=0.5, markerstrokewidth=0.1, ms=10)

    # plot!(part_func[:,1], 2 .* part_func[:,1] .* log(2))
    fig = plot!(xlabel="Number of node p/ partition", ylabel="ln(Z) - Entropy", legend=:topleft)
    savefig(fig, "/home/javier/Projects/RBM/Results/not_so_smallRBMs.png")
end

###################################

begin
    # include("../therm.jl")
    include("../configs/yaml_loader.jl")
    PATH = "/home/javier/Projects/RBM/Results/"
    # dev = gpu
    # β = 1.0
    config, _ = load_yaml_iter();
end
config.model_analysis["files"][1]

modelName = "PCD-FMNIST-500-replica1-L" #config.model_analysis["files"][1]
rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=100);


rbm, J, m, hparams, rbmZ = initModel(nv=1784, nh=1784, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")

@time log_pf_Gauss(J, hparams)
@time AIS(J, hparams, 500, 500, 20)
@time RAIS(J, hparams, 500, 500, 20)
@time log_pf_Gauss_Approx_beta(J, hparams, 0.025)
@time log_pf_Gauss_orthogonal(J, hparams, 0.05)
# @time log_pf_Gauss_orthogonal(J, hparams)
log_configurational_entropy(hparams)

F = LinearAlgebra.svd(J.w, full=true)
plot(F.S, yscale=:linear, color=:red, frame=:box,
    label="Singular Values", s=:auto, markershapes = :star5, lw=0.5, markerstrokewidth=0.1, ms=10)
hline!([4])
minimum(F.S)

plot(sum(F.U, dims=1)[1,:] |> cpu)
plot!(sum(F.U .* (F.U .> 0), dims=1)[1,:] |> cpu)
plot!(sum(F.U .* (F.U .< 0), dims=1)[1,:] |> cpu)

plot(sum(F.V, dims=1)[1,:] |> cpu)
plot!(sum(F.V .* (F.V .> 0), dims=1)[1,:] |> cpu)
plot!(sum(F.V .* (F.V .< 0), dims=1)[1,:] |> cpu)

J.w = F.U * gpu(diagm(F.S .+ 18)) * F.Vt
J.w = F.U * gpu(diagm(4 .* F.S ./ maximum(F.S) )) * F.Vt
J.w = F.U * gpu(diagm(vcat([100,80], ones(782)))) * F.Vt
F.S
J.w = F.U * gpu(diagm((rand(500) .* 5) .+ 15)) * F.Vt
J.w = F.U * gpu(diagm(ones(size(F.S)) .* 1)) * F.Vt


F = LinearAlgebra.svd(J.w, full=true)
#increase temperature
β = 4/√(maximum(F.S)^2)*0.6 
#lower temperature
β = 4/minimum(F.S)*1.4 
J.w = @. J.w*β
J.a = @. J.a*β
J.b = @. J.b*β


begin
    rais, ais, orth_g = [], [], []
    for i in 2:10
        @info i
        F = LinearAlgebra.svd(J.w, full=true)
        J.w = F.U * gpu(diagm(ones(size(F.S)) .* i)) * F.Vt
        push!(ais, AIS(J, hparams, 500, 500, 20))
        push!(rais, RAIS(J, hparams, 500, 500, 20))
    end        
end

plot(2:10, rais)
plot!(2:10, ais/1000)
rais

F = LinearAlgebra.svd(J.w, full=true)
plot()
for i in 2:10
    J.w = F.U * gpu(diagm(ones(size(F.S)) .* i)) * F.Vt
    plot!([log_pf_Gauss_orthogonal(J, hparams, i) for i in 0.1:1:20], label=i, frame=:box,
    s=:auto, markershapes = :auto, lw=0.5, markerstrokewidth=0.1, ms=10)
    hline!([rais[i-1]], lw=2 )
end
plot!(size=(900,600))

plot([log_pf_Gauss_orthogonal(J, hparams, i) for i in 0.1:1:20], label=2, lw=2, ls=:dash)


#kurtosis
config.model_analysis["files"]
begin 
    modelName = config.model_analysis["files"][1]
    rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=100);
    krt_x = Array(zeros(hparams.nv, 5, 100))
    krt_y = Array(zeros(hparams.nh, 5, 100))
    sv = Array(zeros(hparams.nh, 5, 100))
    for i in 1:5
        modelName = config.model_analysis["files"][i+0]
        for ind in 1:100
            # @info i, ind
            rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=ind*1);
            F = LinearAlgebra.svd(J.w, full=true)
            krt_x[:,i, ind] = sum(F.U' |> cpu, dims=2) .^ 2
            krt_y[:,i, ind] = sum(F.Vt |> cpu, dims=2) .^ 2
            sv[:,i, ind] = F.S |> cpu
        end
    end
end
plot(1 .+ 4 .* reshape(mean(krt_x, dims=2),784,100)', ribbon=reshape(std(krt_x, dims=2),784,100)', legend=false, yscale=:log10)
plot(1 .+ 4 .* reshape(mean(krt_y, dims=2),784,100)', ribbon=reshape(std(krt_y, dims=2),784,100)', legend=false, yscale=:log10)

fig1 = plot(1:nv, (1 .+ 4 .* mean(krt_x,dims=2)[:,1,end]) , ribbon=std(krt_x,dims=2)[:,1,end], lw=2, label="x", 
    frame=:box,  markersize=7, markershapes = :circle, markerstrokewidth=0.5, yscale=:log10)

plot(1:nv,  std(krt_x,dims=2), ribbon=std(krt_x,dims=2), lw=2, label="x", 
    frame=:box,  markersize=7, markershapes = :circle, markerstrokewidth=0.5)

plot(log.(1 .+ 4 .* sum(F.U', dims=2) .^ 2), st=:histogram)
plot!(log.(1 .+ 4 .* sum(F.Vt, dims=2) .^ 2), st=:histogram)

begin
    smp = 5000
    nv = 100
    krt_x = CuArray(zeros(nv, smp))
    krt_y = CuArray(zeros(nv, smp))
    sv = CuArray(zeros(nv, smp))
    for i in 1:smp
        rbm, J, m, hparams, rbmZ = initModel(nv=nv, nh=nv, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
        F = LinearAlgebra.svd(J.w, full=true)
        krt_x[:,i] = sum(F.U', dims=2) .^ 2
        krt_y[:,i] = sum(F.Vt, dims=2) .^ 2
        sv[:,i] = F.S
    end
end

fig1 = plot(1:nv, 1 .+ 4 .* mean(krt_x,dims=2), ribbon=std(krt,dims=2), lw=2, label="x", 
    frame=:box,  markersize=7, markershapes = :circle, markerstrokewidth=0.5)
fig1 = plot!(1:nv, 1 .+ 4 .* mean(krt_y,dims=2), ribbon=std(krt,dims=2), lw=2, label="y", 
    xlabel="layer node", ylabel="Kurtosis", markersize=7, markershapes = :circle, markerstrokewidth=0.5, fillalpha=0.3)
    savefig(fig1, "/home/javier/Projects/RBM/Results/Figs/kurtosisOfRandRBMs.png")
fig2 = plot(sv[:][1:100000], st=:histogram, lw=0, normalize=true, xlabel="Singular values", frame=:box, legend=false,
    ylabel="PDF")
savefig(fig2, "/home/javier/Projects/RBM/Results/Figs/svEnsemble.png")



using StatsBase
begin
    smp = 10
    nv = 8
    krt_2 = Array(zeros(nv, smp))
    sd = Array(zeros(nv, smp))
    m_num = Array(zeros(nv, smp))
    m_an = Array(zeros(nv, smp))
    for i in 1:smp
        rbm, J, m, hparams, rbmZ = initModel(nv=8, nh=8, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
        vs = generate_binary_states(size(J.a,1))
        F = LinearAlgebra.svd(J.w, full=true)
        xs = cpu(F.U') * hcat(vs...)
        krt_2[:,i] = [kurtosis(xs[i,:]) for i in 1:8]
        sd[:,i] = [std(xs[i,:]) for i in 1:8]
        m_num[:,i] = [mean(xs[i,:]) for i in 1:8]
        m_an[:,i] = 0.5 * sum(F.U', dims=2)
    end
end
mean(krt_2,dims=2)
plot(1:nv, mean(krt_2, dims=2), ribbon=std(krt_2,dims=2), lw=2)

plot(1:nv, mean(sd, dims=2), ribbon=std(sd,dims=2), lw=2)


plot(1:nv, mean(m_num, dims=2), ribbon=std(m_num,dims=2), lw=2)
plot!(1:nv, mean(m_an, dims=2), ribbon=std(m_an,dims=2), lw=2)


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
    mc_steps=500
    save_every = 1
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

    # f6 = plot([kurtosis(x[i,:,j]) for i in 1:hparams.nv, j in 2:ar_size]', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
    # xlabel="MC step (x $(save_every))", ylabel="kutorsis(x)", frame=:box, opacity=0.5);

    # f7 = plot([kurtosis(y[i,:,j]) for i in 1:hparams.nh, j in 2:ar_size]', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
    # xlabel="MC step (x $(save_every))", ylabel="kutorsis(y)", frame=:box, opacity=0.5);

    plot(f1,f2,f3,f4,f5, size=(1200,900))
end

plot(reshape(mean(x,dims=2),:,ar_size)', color=[RGB(1 - i/sample_size, 0, i/sample_size) for i in 1:sample_size]', legend=:none,
        xlabel="MC step (x $(save_every))", ylabel="⟨x⟩", frame=:box, title=modelName, opacity=0.5);

plot(reshape(std(x,dims=2),:,ar_size)[1:20,1:100]', legend=false)

###########
function gibbs_sample(v, J, β, dev)
    h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * v .+ J.b)))) |> dev
    v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * h .+ J.a)))) |> dev
    return v, h
end

function self_correlation(arr, t, τ)
    μ = mean(arr, dims=1)
    C = mean(arr[:,:,t+τ] .* arr[:,:,t], dims=1) .- μ[:,:,t+τ] .* μ[:,:,t]
    # mean(C, dims=1) / var(arr[:,:,t], dims=1)
    C ./ var(arr[:,:,t], dims=1)
end

function c_i_tau(arr,τ, steps=size(arr,3))
    C = self_correlation(arr, 1, τ)
    CT = C
    for t in 2:steps-τ
        C = self_correlation(arr, t, τ)
        CT = cat([CT, C]..., dims=1)
    end
    mean(CT, dims=1)
end

function generate_samples(num_iterations, num, J, hparams, gpu)
    v_samples = Array{Array{Float32,2}}(undef, num_iterations) 
    h_samples = Array{Array{Float32,2}}(undef, num_iterations)
    v = rand([0,1], hparams.nv, num) |> gpu
    for i in 1:num_iterations
        v, h = gibbs_sample(v, J, 1.0, gpu)
        v_samples[i] = Array(v)
        h_samples[i] = Array(h)
    end
    vs = permutedims(cat(v_samples..., dims=3), [2,1,3])
    hs = permutedims(cat(h_samples..., dims=3), [2,1,3])
    return vs, hs
    
end

function build_correlation_functions(vs, hs, tmax, Δt=1)
    c_dict = Dict()
    for (i,ar) in enumerate([vs, hs])
        @info i
        CT = c_i_tau(ar,0)
        for τ in 1:Δt:tmax-1
            @info τ
            CT = cat([CT,c_i_tau(ar, τ)]..., dims=1) 
        end
        c_dict[i] = CT
    end
    return c_dict
    
end

num_iterations = 900  # replace with the number of iterations you want
num=100
vs,hs = generate_samples(num_iterations, num, J, hparams, gpu)
c_dict = build_correlation_functions(vs, hs, num_iterations, 10)

plot(c_dict[1], legend=false, lw=2, xlabel="Time (x10)", ylabel="Correlation", title="Visible layer")
plot(c_dict[2], legend=false, lw=2, xlabel="Time (x10)", ylabel="Correlation", title="Hidden layer")


# v_samples = Array{Array{Float32,2}}(undef, num_iterations) 
# h_samples = Array{Array{Float32,2}}(undef, num_iterations)
# num=100
# v = rand([0,1], hparams.nv, num) |> gpu
# for i in 1:num_iterations
#     v, h = gibbs_sample(v, J, 1.0, gpu)
#     v_samples[i] = Array(v)
#     h_samples[i] = Array(h)
# end
# vs = permutedims(cat(v_samples..., dims=3), [2,1,3])
# hs = permutedims(cat(h_samples..., dims=3), [2,1,3])

# C = self_correlation(hs, 2, 0)
# CT = c_i_tau(vs,0)

# c_dict = Dict()

# for (i,ar) in enumerate([vs, hs])
#     CT = c_i_tau(ar,0)
#     for τ in 1:40:400
#         @info τ
#         CT = cat([CT,c_i_tau(ar, τ)]..., dims=1) 
#     end
#     c_dict[i] = CT
# end
# c_dict[1]
# plot(c_dict[1])

v = rand([0,1], hparams.nv, num) |> gpu
# rbm.v
num=500
for i in 1:10*num_iterations
    v, h = gibbs_sample(v, J, 1.0, gpu)
end
heatmap(reshape(cpu(v[:,1]),28,28))

lnum=10 #Int(sqrt(sample_size))
mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
f1 = heatmap(cpu(mat_rot), size=(900,900))

v = permutedims(vs[:,:,end], [2,1])

#FMNIST
rbm, J, m, hparams, rbmZ = initModel(nv=28*28, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
x, y = loadData(; hparams, dsName="FMNIST", numbers=collect(0:9), testset=true);

lnum=10 #Int(sqrt(sample_size))
mat = cat([cat([reshape(x[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
f1 = heatmap(cpu(mat_rot), size=(900,900))
