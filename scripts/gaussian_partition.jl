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
    F = LinearAlgebra.svd(J.w, full=true)
    x_m = sum(F.U', dims=2)/2
    x_σ = typeof(J.w)(ones(size(x_m))/2)
    y_m = sum(F.Vt, dims=2)/2
    y_σ = typeof(J.w)(ones(size(y_m))/2)

    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    Z_mf = 2^(hparams.nv+hparams.nh) * prod( @. exp(B^2/(4*A) + C) * (2*x_σ^2*A)^(-1/2))
    Z_mf
end

function log_pf_Gauss(J::Weights, hparams::HyperParams)
    F = LinearAlgebra.svd(J.w, full=true)
    x_m = sum(F.U', dims=2)/2
    x_σ = typeof(J.w)(ones(size(x_m))/2)
    y_m = sum(F.Vt, dims=2)/2
    y_σ = typeof(J.w)(ones(size(y_m))/2)

    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    lnZ_mf = sum( @. log(exp(B^2/(4*A) + C) * (2*x_σ^2*A)^(-1/2))) + (hparams.nv+hparams.nh) * log(2) 
    lnZ_mf
end

# function log_pf_Gauss(J::Weights, hparams::HyperParams, n::Union{Float16, Float32, Float64})
#     """
#         Very unstable
#     """
#     F = LinearAlgebra.svd(J.w, full=true)
#     x_m = sum(F.U', dims=2)/2
#     x_σ = typeof(J.w)(ones(size(x_m))/2)
#     y_m = sum(F.Vt, dims=2)/2
#     y_σ = typeof(J.w)(ones(size(y_m))/2)

#     B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
#     A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
#     C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

#     # lnZ_mf = @. B^2/(4*A) + C - 0.5 * log(2*x_σ^2) + log(f(A,B,x_σ,x_m,1))
#     lnZ_mf = @. B^2/(4*A) + C - 0.5 * log(2π*x_σ^2) 
#     lnZ_mf = lnZ_mf .+ log.(I(A,B,x_m,x_σ,n))
#     # lnZ_mf = log(sum( @. exp(B^2/(4*A) + C) * (2*A)^(-1/2))) + (hparams.nv+hparams.nh) * log(2) 
#     sum(lnZ_mf) + (hparams.nv+hparams.nh) * log(2) 
# end

# function I(A,B,μ,σ, n=1.0)
#     A = cpu(A)
#     B = cpu(B)
#     σ = cpu(σ)
#     μ = cpu(μ)
#     sqrtAcomplex = @. √complex(A)
#     res = @. real(√π/2 * 1/sqrtAcomplex * ( erf(sqrtAcomplex * (μ + σ * n - B/(2*A))) - erf(sqrtAcomplex * (μ - σ * n - B/(2*A))) ))
#     return gpu(res)
# end

function ln_I(A,B,μ, Δσ)
    if A > 0
        return 0.5 * log(π/A)
    else
        return - A * (μ - B/(2*A))^2 + log(2*Δσ)
    end
end

function free_energy(A,B, C,μ, σ, Δσ; THRS=20, THRS_B=20)
    if isapprox(A, 0) || abs(B^2/(4*A)) > THRS
        if abs(B) > THRS_B
        # return B * μ + C + log(2*sinh(B*σ)/B)
            return B * μ + C + B*σ - log(abs(B))
        else
            return B * μ + C + log(2*sinh(B*σ)/B)
        end
    else
        return B^2/(4*A) + C + ln_I(A,B,μ,Δσ)
    end
end

function ln_Z_λ(A,B, C,μ, σ, Δσ; THRS_B=20)
    if isapprox(A, 0)
        if abs(B) > THRS_B
            return B * μ + C + abs(B)*Δσ - log(abs(B)) - 0.5 * log(2*π*σ^2)
        else
            return B * μ + C + log(2*sinh(B*Δσ)/B) - 0.5 * log(2*π*σ^2)
        end
    elseif A > 0
        return B^2/(4*A) + C - 0.5 * log(2 * A * σ^2)
    elseif A < 0
        return abs(A) * μ^2 + B * μ + C + 0.5 * log(2 * Δσ^2 / (π * σ^2))
    end
end


function log_pf_Gauss_Approx(J::Weights, hparams::HyperParams, Δσ::Union{Float16, Float32, Float64})

    F = LinearAlgebra.svd(J.w, full=true)
    x_m = sum(F.U', dims=2)/2
    x_σ = typeof(J.w)(ones(size(x_m))/2)
    y_m = sum(F.Vt, dims=2)/2
    y_σ = typeof(J.w)(ones(size(y_m))/2)

    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    Δσ = typeof(A)(ones(size(A)) * Δσ)
    lnZ_mf = @. B^2/(4*A) + C - 0.5 * log(2π*x_σ^2) + ln_I(A,B,x_m,Δσ)
    sum(lnZ_mf) + (hparams.nv+hparams.nh) * log(2) 
end

function log_pf_Gauss_Approx_beta(J::Weights, hparams::HyperParams, Δσ::Union{Float16, Float32, Float64}; THRS=10, THRS_B=20)

    F = LinearAlgebra.svd(J.w, full=true)
    x_m = sum(F.U', dims=2)/2
    x_σ = typeof(J.w)(ones(size(x_m))/2)
    y_m = sum(F.Vt, dims=2)/2
    y_σ = typeof(J.w)(ones(size(y_m))/2)
    a = F.U' * J.a
    b = F.Vt * J.b

    B = y_m ./ (y_σ .^ 2) .+ F.S .* ( x_σ .^ 2 .* F.U' * J.a .+ x_m ) .+ F.Vt *J.b
    A = 1 ./ (2 .* y_σ .^ 2) - x_σ .^ 2 .* F.S .^ 2 ./ 2
    C = .- y_m .^ 2 ./ (2 .* y_σ .^ 2) .+ x_m .* F.U' * J.a .+ x_σ .^ 2 .* (F.U' * J.a) .^ 2  ./ 2

    Δσ = typeof(A)(ones(size(A)) * Δσ)
    #################
    A = cpu(A)
    B = cpu(B)
    C = cpu(C)
    x_m = cpu(x_m)
    x_σ = cpu(x_σ)
    Δσ = cpu(Δσ)
    a = cpu(a)
    b = cpu(b)
    ################
    ln_Z_ϵ = (hparams.nv+hparams.nh) * log(2) 
    # ln_Z_unc = sum(@. β * x_m * a + β^2 * σ^2 * a^2/2)
    ln_Z_sv = @. ln_Z_λ(A,B, C,x_m, x_σ, Δσ)
    ln_Z = ln_Z_ϵ + sum(ln_Z_sv)
    # F_E_term = @. free_energy(A,B, C,x_m, x_σ, Δσ, THRS=THRS, THRS_B=THRS_B)
    # lnZ_mf = @. - 0.5 * log(2π*x_σ^2) + F_E_term
    # sum(lnZ_mf) + (hparams.nv+hparams.nh) * log(2) 
    ln_Z
end