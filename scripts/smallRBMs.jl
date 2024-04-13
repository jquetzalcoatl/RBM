begin
    using CUDA, Flux, HDF5
    using Base.Threads
    using StatsPlots
    CUDA.device_reset!()
    CUDA.device!(0)
    Threads.nthreads()
end

include("../utils/train.jl")

Random.seed!(1234);
d = Dict("bw"=>false)
rbm, J, m, hparams, opt = train(d, epochs=50, nv=28*28, nh=500, batch_size=500, lr=0.0001, t=100, plotSample=true, 
    annealing=false, learnType="CD", β=1, β2 = 1, gpu_usage = false, t_samp = 100, num=100, optType="Adam", numbers=[1,5], 
    savemodel=false, snapshot=1)

rbm, J, m, hparams, rbmZ = initModel(nv=10, nh=5, batch_size=500, lr=1.5, t=10, gpu_usage = false, optType="Adam")
# opt = initOptW(hparams, J);



function generate_binary_states(N::Int)
    return [digits(i, base=2, pad=N) for i in 0:(2^N - 1)]
end

### Gaussian Approximation
function pf_Gauss(N::Int)
    rbm, J, m, hparams, rbmZ = initModel(nv=N, nh=N, batch_size=1, lr=1.5, t=10, gpu_usage = false, optType="Adam")
    vs = generate_binary_states(hparams.nv)
    hs = generate_binary_states(hparams.nh)

    F = LinearAlgebra.svd(J.w, full=true)
    xs = F.U' * hcat(vs...)
    ys = F.Vt * hcat(hs...)

    x_m, x_σ = mean(xs, dims=2)[:], std(xs, dims=2)[:]
    y_m, y_σ = mean(ys, dims=2)[:], std(ys, dims=2)[:]

    # B = y_m ./ (y_σ .^ 2) .* F.S .* ( x_σ .^ 2 .* F.U' * J.a .- x_m )
    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    # C = 1 ./ (2 .* y_σ .^ 2) .* ( y_m .^ 2 .- x_σ .^ 2 .* y_σ .^ 2 .* (F.U' * J.a) .^ 2 .+ 2 .* x_m .* F.U' * J.a .* y_σ .^ 2)
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    # Z_mf = 2^(2*N) * sum( @. exp(B^2/(2*A) - C) * (1 - x_σ^2 * F.S * y_σ^2)^(-1/2))
    Z_mf = 2^(2*N) * sum( @. exp(B^2/(4*A) + C) * (2*A)^(-1/2))
    Z_mf
end

function pf_Gauss(J::Weights, hparams::HyperParams)
    # rbm, J, m, hparams, rbmZ = initModel(nv=N, nh=N, batch_size=1, lr=1.5, t=10, gpu_usage = false, optType="Adam")
    vs = generate_binary_states(hparams.nv)
    hs = generate_binary_states(hparams.nh)

    F = LinearAlgebra.svd(J.w, full=true)
    xs = F.U' * hcat(vs...)
    ys = F.Vt * hcat(hs...)

    x_m, x_σ = mean(xs, dims=2)[:], std(xs, dims=2)[:]
    y_m, y_σ = mean(ys, dims=2)[:], std(ys, dims=2)[:]

    # B = y_m ./ (y_σ .^ 2) .* F.S .* ( x_σ .^ 2 .* F.U' * J.a .- x_m )
    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    # C = 1 ./ (2 .* y_σ .^ 2) .* ( y_m .^ 2 .- x_σ .^ 2 .* y_σ .^ 2 .* (F.U' * J.a) .^ 2 .+ 2 .* x_m .* F.U' * J.a .* y_σ .^ 2)
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    # Z_mf = 2^(hparams.nv+hparams.nh) * sum( @. exp(B^2/(2*A) - C) * (1 - x_σ^2 * F.S^2 * y_σ^2)^(-1/2))
    Z_mf = 2^(hparams.nv+hparams.nh) * sum( @. exp(B^2/(4*A) + C) * (2*A)^(-1/2))
    Z_mf
end

begin
    replicas = 5
    part_func = zeros(1,replicas+1)
    part_func_G = zeros(1,replicas+1)
    for s in 2:15
        vs = generate_binary_states(s)
        hs = generate_binary_states(s)
        zs = []
        zs_G = []
        for r in 1:replicas
            rbm, J, m, hparams, rbmZ = initModel(nv=s, nh=s, batch_size=1, lr=1.5, t=10, gpu_usage = false, optType="Adam")

            z = sum([exp(v' * J.a + h' * J.b + v' * J.w * h) for v in vs, h in hs])
            push!(zs,z)

            push!(zs_G, pf_Gauss(J, hparams))
        end
        # @info size(zs), vcat(s,zs)
        part_func = vcat(part_func, hcat(s,zs'))
        part_func_G = vcat(part_func_G, hcat(s,zs_G'))
    end

end

begin
    # F = -kTln(Z)
    # F = E - TS
    # ln(Z) - S = - F/kT - S = - E/kT 
    plot(part_func[:,1], log.(mean(part_func[:,2:end], dims=2)[:]) .- 2 .* part_func[:,1] .* log(2), 
        yerr=log.(std(part_func[:,2:end], dims=2)[:]), frame=:box,
        label="Exact", s=:auto, markershapes = :circle, lw=0.5, markerstrokewidth=0.1, ms=10)
    plot!(part_func_G[:,1], log.(mean(part_func_G[:,2:end], dims=2)[:]) .- 2 .* part_func[:,1] .* log(2), 
        yerr=log.(std(part_func_G[:,2:end], dims=2)[:]), color=:red, frame=:box,
        label="Approximation", s=:auto, markershapes = :auto, lw=0.5, markerstrokewidth=0.1, ms=10)
    # plot!(part_func[:,1], 2 .* part_func[:,1] .* log(2))
    plot!(xlabel="Number of nodes", ylabel="ln(Z) - Entropy", legend=:topleft)
end


part_func[:,2] ./ (2 .^ (2 .* part_func[:,1]))

begin
    N=25
    rbm, J, m, hparams, rbmZ = initModel(nv=N, nh=N, batch_size=1, lr=1.5, t=10, gpu_usage = false, optType="Adam")
    vs = generate_binary_states(hparams.nv)
    hs = generate_binary_states(hparams.nh)

    F = LinearAlgebra.svd(J.w, full=true)
    xs = F.U' * hcat(vs...)
    ys = F.Vt * hcat(hs...)

    x_m, x_σ = mean(xs, dims=2)[:], std(xs, dims=2)[:]
    y_m, y_σ = mean(ys, dims=2)[:], std(ys, dims=2)[:]

    
    # B = y_m ./ (y_σ .^ 2) .* F.S .* ( x_σ .^ 2 .* F.U' * J.a .- x_m )
    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    # C = 1 ./ (2 .* y_σ .^ 2) .* ( y_m .^ 2 .- x_σ .^ 2 .* y_σ .^ 2 .* (F.U' * J.a) .^ 2 .+ 2 .* x_m .* F.U' * J.a .* y_σ .^ 2)
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    # Z_mf = 2^(hparams.nv+hparams.nh) * sum( @. exp(B^2/(2*A) - C) * (1 - x_σ^2 * F.S^2 * y_σ^2)^(-1/2))
    Z_mf = 2^(hparams.nv+hparams.nh) * sum( @. exp(B^2/(4*A) + C) * (2*A)^(-1/2))
end

begin
    idx=1
    plot(xs[idx,:], st=:histogram, normalize=true, bins=200, label="data")
    plot!(-2:0.1:3, x->1/√(2π*x_σ[idx]^2) * exp(-(x-x_m[idx])^2/(2*x_σ[idx]^2)), lw=2, label="Gaussian")
end

begin
    idx=1
    plot(ys[idx,:], st=:histogram, normalize=true, bins=200, label="data")
    plot!(-2:0.01:3, x->1/√(2π*y_σ[idx]^2) * exp(-(x-y_m[idx])^2/(2*y_σ[idx]^2)), lw=2, label="Gaussian")
end

using StatsBase

xs
plot([kurtosis(xs[i,:]) for i in 1:N], frame=:box,
    s=:auto, markershapes = :circle, lw=0.5, markerstrokewidth=0.1, ms=5)


sum(F.U', dims=2)/2
x_m

sum(F.U' .^ 2, dims=2)/2
x_σ

#Do AIS and RAIS to estimate Z and compare with Gaussian method