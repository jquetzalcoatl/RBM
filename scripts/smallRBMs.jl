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


modelName = config.model_analysis["files"][14]
rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=100);

rbm, J, m, hparams, rbmZ = initModel(nv=784, nh=500, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")

@time log_pf_Gauss(J, hparams)
@time AIS(J, hparams, 500, 500, 20)
@time RAIS(J, hparams, 500, 500, 20)
@time log_pf_Gauss_Approx_beta(J, hparams, 0.5)
@time log_pf_Gauss_orthogonal(J, hparams, 0.5)
@time log_pf_Gauss_orthogonal(J, hparams)
log_configurational_entropy(hparams)

F = LinearAlgebra.svd(J.w, full=true)
plot(F.S, yscale=:linear, color=:red, frame=:box,
    label="Singular Values", s=:auto, markershapes = :star5, lw=0.5, markerstrokewidth=0.1, ms=10)
hline!([4])
minimum(F.S)


J.w = F.U * gpu(diagm(F.S .+ 20)) * F.Vt
J.w = F.U * gpu(diagm(F.S ./ maximum(F.S) )) * F.Vt
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
plot!(sum(F.U', dims=2) .^ 2, yscale=:log10)
sum(F.U', dims=2) .^ 4

begin
    smp = 1000
    nv = 1000
    krt = CuArray(zeros(nv, smp))
    sv = CuArray(zeros(nv, smp))
    for i in 1:smp
        rbm, J, m, hparams, rbmZ = initModel(nv=nv, nh=nv, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
        F = LinearAlgebra.svd(J.w, full=true)
        krt[:,i] = sum(F.U', dims=2) .^ 2
        sv[:,i] = F.S
    end
end

plot(1:nv, 1 .+ 4 .* mean(krt,dims=2), ribbon=std(krt,dims=2), lw=2)
plot(sv[:], st=:histogram)


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
