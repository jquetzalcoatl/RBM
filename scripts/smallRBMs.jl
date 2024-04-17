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
include("../scripts/RAIS.jl")

Random.seed!(1234);
# d = Dict("bw"=>false)
# rbm, J, m, hparams, opt = train(d, epochs=50, nv=28*28, nh=500, batch_size=500, lr=0.0001, t=100, plotSample=true, 
    # annealing=false, learnType="CD", β=1, β2 = 1, gpu_usage = false, t_samp = 100, num=100, optType="Adam", numbers=[1,5], 
    # savemodel=false, snapshot=1)

rbm, J, m, hparams, rbmZ = initModel(nv=5, nh=5, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
# opt = initOptW(hparams, J);

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
    # part_func = part_func[2:end,:]
    # part_func_G = part_func_G[2:end,:]

    # part_func_AIS = part_func_AIS[2:end,:]
    # part_func_RAIS = part_func_RAIS[2:end,:]

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
   # replicas = 50
#    part_func = zeros(max_size-2+1,replicas+1)
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

        #    push!(zs,partition_function(J))

           push!(zs_G, log_pf_Gauss_beta(J, hparams))
       end

    #    part_func[i,:] = hcat(s,zs') #vcat(part_func, hcat(s,zs'))
       part_func_G[i,:] = hcat(s,zs_G') #vcat(part_func_G, hcat(s,zs_G'))

       part_func_AIS[i,:] = hcat(s,AIS(J, hparams, samples, mcs, nbeta)) #vcat(part_func_AIS, hcat(s,AIS(J, hparams, samples, mcs, nbeta)))
       part_func_RAIS[i,:] = hcat(s,RAIS(J, hparams, samples, mcs, nbeta)) #vcat(part_func_RAIS, hcat(s,RAIS(J, hparams, samples, mcs, nbeta)))
   end
   # part_func = part_func[2:end,:]
   # part_func_G = part_func_G[2:end,:]

   # part_func_AIS = part_func_AIS[2:end,:]
   # part_func_RAIS = part_func_RAIS[2:end,:]

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


modelName = config.model_analysis["files"][12]
rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=1);

rbm, J, m, hparams, rbmZ = initModel(nv=10, nh=10, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")

@time log_pf_Gauss_beta(J, hparams)
@time AIS(J, hparams, 500, 500, 20)
@time RAIS(J, hparams, 500, 500, 20)



F = LinearAlgebra.svd(J.w, full=true)

x_m, x_σ = sum(F.U', dims=2)/2, sum(F.U' .^ 2, dims=2)/2

y_m, y_σ = sum(F.Vt, dims=2)/2, sum(F.Vt .^ 2, dims=2)/2

B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2


A = cpu(A)
B = cpu(B)
x_σ = cpu(x_σ)
x_m = cpu(x_m)
@. √(1/abs(A)) * ( (A > 0) + (A <= 0) * 0.5 * (erfi(√abs(A) * (x_σ + B/(2*A) - x_m) ) + erfi(√abs(A) * (x_σ - B/(2*A) + x_m) )) )

sqrt.(complex(A))

(erfi(√abs(A) * (x_σ + B/(2*A) - x_m) ) + erfi(√abs(A) * (x_σ - B/(2*A) + x_m) ))

√complex(-1)
@. (A <= 0) * 0.5 * (erfi(√abs(A) * (x_σ + B/(2*A) - x_m) ) + erfi(√abs(A) * (x_σ - B/(2*A) + x_m) ))

@. √abs(A) * (x_σ + B/(2*A) - x_m)
@. √abs(A) * (x_σ - B/(2*A) + x_m) 
A

erfi(4)
-im*erf(im*10)
1+2i

erf(100)


function I(A,B,μ,σ)
    A = cpu(A)
    B = cpu(B)
    σ = cpu(σ)
    μ = cpu(μ)
    sqrtAcomplex = @. √complex(A)
    res = @. √π/2 * 1/sqrtAcomplex * ( erf(sqrtAcomplex * (μ - σ - B/(2*A))) - erf(sqrtAcomplex * (- μ + σ - B/(2*A))) )
    return gpu(res)
end

plot(real(I(A,B, x_m, x_σ ./ 2)))

