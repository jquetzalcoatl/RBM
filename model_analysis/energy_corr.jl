begin
    using CUDA, Flux, HDF5
    using Base.Threads
    using StatsPlots
    using BSplineKit
    using QuadGK
    using FFTW
    CUDA.device_reset!()
    CUDA.device!(2)
    Threads.nthreads()
end

include("../utils/train.jl")

include("../scripts/exact_partition.jl")
include("../scripts/gaussian_partition.jl")
include("../scripts/gaussian_orth_partition.jl")
include("../scripts/RAIS.jl")



# Random.seed!(1234);
# rbm, J, m, hparams, rbmZ = initModel(nv=5, nh=5, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")

begin
    # include("../therm.jl")
    include("../configs/yaml_loader.jl")
    PATH = "/home/javier/Projects/RBM/Results/"
    # dev = gpu
    # β = 1.0
    config, _ = load_yaml_iter();
end
config.model_analysis["files"]

# modelName = "PCD-FMNIST-500-replica1-L" #config.model_analysis["files"][1]
modelName = "CD-FMNIST-500-T1000-BW-replica1-L"
modelName = config.model_analysis["files"][1]

rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=100);
# PATH="/home/javier/Projects/RBM/NewResults/$modelName/"
# isdir(PATH) ? (@info "Directory exists") : mkdir(PATH)

rbm, J, m, hparams, rbmZ = initModel(nv=10, nh=10, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
######
l=10
J.w = rand(l,l) .* 2 .- 1 |> gpu
J.a = ones(l) |> gpu #rand(l) .* 2 .- 1 |> gpu
J.b = ones(l) |> gpu #rand(l) .* 2 .- 1 |> gpu
######

function gibbs_sample(v, J, dev, num, β=1)
    h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* ( J.w' * v .+ J.b )))) |> dev
    v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* ( J.w * h .+ J.a)))) |> dev
    return v, h
end

function obtain_rel_error(rbm,J,dev; num=500, steps=1000, mode="avg")
    if mode == "exact"
        if size(rbm.v,1) + size(rbm.h,1) > 40
            @warn "The RBM size is too big to compute the exact partition"
            return 0,0,0
        end
        rbm.v=hcat(generate_binary_states(hparams.nv)...) |> dev
        rbm.h=hcat(generate_binary_states(hparams.nh)...) |> dev

        Z = partition_function(J)
        dEdaE = -rbm.v * (H(rbm,J) .* exp.(-H(rbm,J)) ./ Z)
        dEda = - rbm.v * (exp.(-H(rbm,J)) ./ Z)
        E = H(rbm,J)' * exp.(-H(rbm,J)) / Z

        dEdbE = -rbm.h * (H(rbm,J) .* exp.(-H(rbm,J)) ./ Z)
        dEdb = - rbm.h * (exp.(-H(rbm,J)) ./ Z)

        dEdJE = - rbm.v * gpu(diagm(H(rbm,J) .* exp.(-H(rbm,J)) ./ Z)) * rbm.h'
        dEdJ = - rbm.v * gpu(diagm(exp.(-H(rbm,J)) ./ Z)) * rbm.h'

        rel_a = (dEdaE .- dEda * E) ./ (dEda )
        rel_b = (dEdbE .- dEdb * E) ./ (dEdb )
        rel_J = (dEdJE .- dEdJ * E) ./ (dEdJ )

    elseif mode == "avg"
        rbm.v = rand([0,1], hparams.nv, num) |> dev
        for i in 1:steps
            rbm.v, rbm.h = gibbs_sample(rbm.v, J, dev, num, 1 - 1/i + 1/steps)
        end
        ϵ = eps(eltype(rbm.v * H(rbm,J)))
        dEdaE = -rbm.v * H(rbm,J) / num
        dEda = -reshape(mean(rbm.v, dims=2),:)
        E =  mean(H(rbm,J))

        dEdbE = -rbm.h * H(rbm,J) / num
        dEdb = - reshape(mean(rbm.h, dims=2),:)

        dEdJE = - rbm.v * gpu(diagm(H(rbm,J))) * rbm.h' / num
        dEdJ = - rbm.v * rbm.h' / num

        rel_a = (dEdaE .- dEda * E) ./ (dEda .+ ϵ)
        rel_b = (dEdbE .- dEdb * E) ./ (dEdb .+ ϵ)
        rel_J = (dEdJE .- dEdJ * E) ./ (dEdJ .+ ϵ)
    elseif mode == "boltzmann"
        if size(rbm.v,1) + size(rbm.h,1) > 40
            @warn "The RBM size is too big to compute avg with Boltzmann factors"
            return 0,0,0
        end
        rbm.v = rand([0,1], hparams.nv, num) |> dev
        for i in 1:steps
            rbm.v, rbm.h = gibbs_sample(rbm.v, J, dev, num, 1)
        end
        ϵ = eps(eltype(rbm.v * H(rbm,J)))
        Z = sum(exp.(-H(rbm,J)))
        dEdaE = -rbm.v * (H(rbm,J) .* exp.(-H(rbm,J)) ./ Z)
        dEda = - (rbm.v * (exp.(-H(rbm,J)) ./ Z))
        E = H(rbm,J)' * exp.(-H(rbm,J)) / Z

        dEdbE = -rbm.h * (H(rbm,J) .* exp.(-H(rbm,J)) ./ Z)
        dEdb = - rbm.h * (exp.(-H(rbm,J)) ./ Z)

        dEdJE = - rbm.v * gpu(diagm(H(rbm,J) .* exp.(-H(rbm,J)) ./ Z)) * rbm.h'
        dEdJ = - rbm.v * gpu(diagm(exp.(-H(rbm,J)) ./ Z)) * rbm.h'

        rel_a = (dEdaE .- dEda * E) ./ (dEdaE .+ ϵ)
        rel_b = (dEdbE .- dEdb * E) ./ (dEdb .+ ϵ)
        rel_J = (dEdJE .- dEdJ * E) ./ (dEdJ .+ ϵ)
    end

    return rel_a, rel_b, rel_J
end

@time rel_a, rel_b, rel_J = obtain_rel_error(rbm,J,gpu, steps=10000, num=10000, mode="exact")
@time rel_a, rel_b, rel_J = obtain_rel_error(rbm,J,gpu, steps=10000, num=1000, mode="boltzmann")
@time rel_a, rel_b, rel_J = obtain_rel_error(rbm,J,gpu, steps=10000, num=10000, mode="avg")

begin
    rel_a_idx = abs.(rel_a) .> 0
    p1=plot(rel_a[rel_a_idx], st=:histogram, label=" a: non-zero vals = $(sum(rel_a_idx)/size(rel_a,1))")
    rel_b_idx = abs.(rel_b) .> 0
    p2=plot(rel_b[rel_b_idx], st=:histogram, bins=30, label=" b: non-zero vals = $(sum(rel_b_idx)/size(rel_b,1))")
    rel_J_idx = abs.(rel_J) .> 0
    p3=plot(rel_J[rel_J_idx], st=:histogram, bins=30, label=" J: non-zero vals = $(sum(rel_J_idx)/prod(size(rel_J)))")
    plot(p1,p2,p3, size=(800,800), lw=0, title="Trained RBM in MNIST w/ PCD")
end

begin
    rel_a_idx = abs.(rel_a) .> 0
    p1=plot(rel_a, st=:histogram, bins=30, label=" a: non-zero vals = $(sum(rel_a_idx)/size(rel_a,1))")
    rel_b_idx = abs.(rel_b) .> 0
    p2=plot(rel_b, st=:histogram, bins=30, label=" b: non-zero vals = $(sum(rel_b_idx)/size(rel_b,1))")
    rel_J_idx = abs.(rel_J) .> 0
    p3=plot(reshape(rel_J,:), st=:histogram, bins=30, label=" J: non-zero vals = $(sum(rel_J_idx)/prod(size(rel_J)))")
    plot(p1,p2,p3, size=(800,800), lw=0.1, title="Untrained RBM", xlabel="ϵ", ylabel="Histogram")
end

H(rbm,J)

sum(H(rbm,J) .* exp.(-H(rbm,J)) ./ partition_function(J))
plot(H(rbm,J), st=:histogram, normalize=true, label="Exact", lw=0)
plot!(H(rbm,J), st=:histogram, normalize=true, label="BGS", bins=50, lw=0, alpha=0.5)
plot!(xlabel="E", ylabel="Histogram")


mean(J.w)
std(J.w)
1/std(J.w)/sqrt(2*l)
mean(J.w)/std(J.w) * sqrt(2*l)

all_sampled_states = vcat([rbm.v, rbm.h]...) |> cpu
uniV = unique(all_sampled_states, dims=2) |> cpu
deg = [size(findall(x->x==uniV[:,j], [all_sampled_states[:,i] for i in 1:size(all_sampled_states,2)]),1) for j in 1:size(uniV,2)]

sum(deg)

plot(sort(deg) ./ 10000, lw=3, xlabel="state index", ylabel="density", label="Samples degeneracy", c=:magenta)
hline!([1/size(uniV,2)], label="Uniform degeneracy", lw=3, c=:black)
sort(deg)
sortperm(deg)
uniV[:,sortperm(deg)]
rbmZ.v = uniV[:,sortperm(deg)][1:4,:] |> gpu
rbmZ.h = uniV[:,sortperm(deg)][5:8,:] |> gpu

bar!(twinx(), cpu(H(rbmZ,J)), label="E", ylabel="Energy", lw=0)

####################
@time rel_a, rel_b, rel_J = obtain_rel_error(rbm,J,gpu, steps=10000, num=10000, mode="exact")
mean_e = sum(H(rbm,J) .* exp.(-H(rbm,J))) / sum(exp.(-H(rbm,J)))
std_e = sum((H(rbm,J) .- mean_e) .^2 .* exp.(-H(rbm,J))) / sum(exp.(-H(rbm,J)))
plot(1 ./ e_list[2:end,1], e_list[2:end,2], lw=2, ribbon=e_list[2:end,3], label="BGS")
hline!([mean_e], lw=2, label="exact", xlabel="Temperature", ylabel="<E>")

e_list = zeros(5,3)
for (i,β) in enumerate([0.005, 0.01, 0.1, 0.5, 1.0])
    num=10000
    rbm.v = rand([0,1], hparams.nv, num) |> gpu
    for i in 1:10000
        rbm.v, rbm.h = gibbs_sample(rbm.v, J, gpu, num, β)
    end
    mean_e = sum(H(rbm,J) .* exp.(-H(rbm,J))) / sum(exp.(-H(rbm,J)))
    std_e = sum((H(rbm,J) .- mean_e) .^2 .* exp.(-H(rbm,J))) / sum(exp.(-H(rbm,J)))
    e_list[i,:] = [β mean_e std_e]
end
e_list

sum(H(rbm,J) .* exp.(-H(rbm,J))) / sum(exp.(-H(rbm,J))) + log(sum(exp.(-H(rbm,J))))
sum(H(rbm,J) .* exp.(-H(rbm,J))) / sum(exp.(-H(rbm,J)))

sum(H(rbm,J) .* exp.(-H(rbm,J))) / sum(exp.(-H(rbm,J))) + log(sum(exp.(-H(rbm,J))))
sum(H(rbm,J) .* exp.(-H(rbm,J))) / sum(exp.(-H(rbm,J)))
