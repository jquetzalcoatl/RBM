using Statistics, LinearAlgebra
using SpecialFunctions
include("../utils/init.jl")

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

    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    Z_mf = 2^(2*N) * prod( @. exp(B^2/(4*A) + C) * (2*A)^(-1/2))
    Z_mf
end

function pf_Gauss(J::Weights, hparams::HyperParams)
    vs = generate_binary_states(hparams.nv)
    hs = generate_binary_states(hparams.nh)

    F = LinearAlgebra.svd(J.w, full=true)
    xs = F.U' * hcat(vs...)
    ys = F.Vt * hcat(hs...)

    x_m, x_σ = mean(xs, dims=2)[:], std(xs, dims=2)[:]
    y_m, y_σ = mean(ys, dims=2)[:], std(ys, dims=2)[:]

    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    Z_mf = 2^(hparams.nv+hparams.nh) * prod( @. exp(B^2/(4*A) + C) * (2*x_σ^2*A)^(-1/2))
    Z_mf
end

function pf_Gauss_beta(J::Weights, hparams::HyperParams)
    # vs = generate_binary_states(hparams.nv)
    # hs = generate_binary_states(hparams.nh)

    F = LinearAlgebra.svd(J.w, full=true)
    # xs = F.U' * hcat(vs...)
    # ys = F.Vt * hcat(hs...)

    # x_m, x_σ = mean(xs, dims=2)[:], std(xs, dims=2)[:]
    x_m, x_σ = sum(F.U', dims=2)/2, sum(F.U' .^ 2, dims=2)/2
    # y_m, y_σ = mean(ys, dims=2)[:], std(ys, dims=2)[:]
    y_m, y_σ = sum(F.Vt, dims=2)/2, sum(F.Vt .^ 2, dims=2)/2

    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    # Z_mf = 2^(hparams.nv+hparams.nh) * sum( @. exp(B^2/(4*A) + C) * (2*A)^(-1/2))
    Z_mf = 2^(hparams.nv+hparams.nh) * prod( @. exp(B^2/(4*A) + C) * (2*x_σ^2*A)^(-1/2))
    Z_mf
end

function log_pf_Gauss_beta(J::Weights, hparams::HyperParams)
    # vs = generate_binary_states(hparams.nv)
    # hs = generate_binary_states(hparams.nh)

    F = LinearAlgebra.svd(J.w, full=true)
    # xs = F.U' * hcat(vs...)
    # ys = F.Vt * hcat(hs...)

    # x_m, x_σ = mean(xs, dims=2)[:], std(xs, dims=2)[:]
    x_m, x_σ = sum(F.U', dims=2)/2, sum(F.U' .^ 2, dims=2)/2
    # y_m, y_σ = mean(ys, dims=2)[:], std(ys, dims=2)[:]
    y_m, y_σ = sum(F.Vt, dims=2)/2, sum(F.Vt .^ 2, dims=2)/2

    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    lnZ_mf = sum( @. log(exp(B^2/(4*A) + C) * (2*x_σ^2*A)^(-1/2))) + (hparams.nv+hparams.nh) * log(2) 
    lnZ_mf
end

function log_pf_Gauss(J::Weights, hparams::HyperParams)

    F = LinearAlgebra.svd(J.w, full=true)

    x_m, x_σ = sum(F.U', dims=2)/2, sum(F.U' .^ 2, dims=2)/2

    y_m, y_σ = sum(F.Vt, dims=2)/2, sum(F.Vt .^ 2, dims=2)/2

    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    lnZ_mf = @. B^2/(4*A) + C - 0.5 * log(2*x_σ^2) + log(f(A,B,x_σ,x_m,1))
    # lnZ_mf = log(sum( @. exp(B^2/(4*A) + C) * (2*A)^(-1/2))) + (hparams.nv+hparams.nh) * log(2) 
    sum(lnZ_mf) + (hparams.nv+hparams.nh) * log(2) 
end

function f(A,B,σ,μ,n=1)
    A = cpu(A)
    B = cpu(B)
    σ = cpu(σ)
    μ = cpu(μ)
    res = @. √(1/A) * ( (A > 0) + (A <= 0) * 0.5 * (erfi(√abs(A) * (σ + B/(2*A) - μ) ) + erfi(√abs(A) * (σ - B/(2*A) + μ) )) )
    return gpu(res)
    # # if A > 0
    #     res = @. √(1/A)
    #     return gpu(res)
    # else
        
    #     res = @. √(1/A) * 0.5 * (erfi(√abs(A) * (σ + B/(2*A) - μ) ) + erfi(√abs(A) * (σ - B/(2*A) + μ) ))
    #     return gpu(res)
    # end
end