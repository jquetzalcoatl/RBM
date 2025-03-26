using Statistics, LinearAlgebra
using SpecialFunctions
include("../utils/init.jl")


function log_pf_Gauss_orthogonal(J::Weights, hparams::HyperParams, Δσ::Union{Float16, Float32, Float64}; THRS=10, β=1.0)

    F = LinearAlgebra.svd(J.w, full=true)
    x_m = sum(F.U', dims=2)/2
    x_σ = typeof(J.w)(ones(size(x_m))/2)
    y_m = sum(F.Vt, dims=2)/2
    y_σ = typeof(J.w)(ones(size(y_m))/2)
    a = F.U' * J.a
    b = F.Vt * J.b

    x_sp = @. - b / F.S
    y_sp = @. - a / F.S
    ϵ = @. a * b / F.S

    u_m = @. 1/√2 * (x_m + y_m - x_sp - y_sp)
    w_m = @. 1/√2 * ( - x_m + y_m + x_sp - y_sp)

    u_σ = typeof(J.w)(ones(size(u_m))/2)
    w_σ = typeof(J.w)(ones(size(w_m))/2)

    λ = F.S

    Δσ = typeof(u_m)(ones(size(u_m)) * Δσ)
    # #################
    λ = cpu(λ)
    w_m = cpu(w_m)
    w_σ = cpu(w_σ)
    u_m = cpu(u_m)
    u_σ = cpu(u_σ)
    Δσ = cpu(Δσ)
    # ################

    lnZw = @. log_Z_λ_w(λ, w_m, w_σ, β)
    lnZu = @. log_Z_λ_u(λ, u_m, u_σ, β, Δσ, THRS=THRS)

    return log_sp_energy(ϵ, β) + sum(lnZw) + sum(lnZu) + log_configurational_entropy(hparams)
end

# function log_pf_Gauss_orthogonal(J::Weights, hparams::HyperParams, Δσ::Union{Float16, Float32, Float64}; THRS=10, β=1.0)

#     F = LinearAlgebra.svd(J.w, full=true)
#     x_m = sum(F.U', dims=2)/2
#     x_σ = typeof(J.w)(ones(size(x_m))/2)
#     y_m = sum(F.Vt, dims=2)/2
#     y_σ = typeof(J.w)(ones(size(y_m))/2)
#     a = F.U' * J.a
#     b = F.Vt * J.b

#     x_sp = @. - b / F.S
#     y_sp = @. - a / F.S
#     ϵ = @. a * b / F.S

#     u_m = @. 1/√2 * (x_m + y_m - x_sp - y_sp)
#     w_m = @. 1/√2 * ( - x_m + y_m + x_sp - y_sp)

#     u_σ = typeof(J.w)(ones(size(u_m))/2)
#     w_σ = typeof(J.w)(ones(size(w_m))/2)

#     λ = F.S

    
#     # #################
#     λ = cpu(λ)
#     w_m = cpu(w_m)
#     w_σ = cpu(w_σ)
#     u_m = cpu(u_m)
#     u_σ = cpu(u_σ)
#     # Δσ = cpu(Δσ)
#     Δσ = (1 ./ .√λ) * Δσ
#     # ################

#     lnZw = @. log_Z_λ_w(λ, w_m, w_σ, β)
#     lnZu = @. log_Z_λ_u(λ, u_m, u_σ, β, Δσ, THRS=THRS)

#     return log_sp_energy(ϵ, β) + sum(lnZw) + sum(lnZu) + log_configurational_entropy(hparams)
# end

log_configurational_entropy(hparams::HyperParams) = (hparams.nv + hparams.nh) * log(2)

log_sp_energy(ϵ, β) = - sum(β * ϵ)

function log_Z_λ_w(λ, ξ, θ, β)
    return - ξ^2/2 * (β * λ * θ^2)/(1+β * λ * θ^2) - 0.5 * log(1+ β * λ * θ^2)
end

function log_Z_λ_u(λ, μ, σ, β, Δσ; THRS=20)
    if isapprox(λ, 1/(β*σ^2))
        if abs(μ*Δσ/σ^2) > THRS
            return 0.5 * log(σ^2/2*π) + μ^2/(2*σ^2) + abs(μ)*Δσ/σ^2 - log(abs(μ))
        else
            return 0.5 * log(σ^2/2*π) + μ^2/(2*σ^2) + log(2*sinh(μ*Δσ/σ^2)/μ)
        end
    elseif λ < 1/(β*σ^2)
        return μ^2/2 * (β*λ*σ^2)/(1-β * λ * σ^2) - 0.5 * log(1 - β * λ * σ^2)
    elseif λ > 1/(β*σ^2)
        return μ^2*λ*β/2 + 0.5 * log(2 * Δσ^2 / (π * σ^2))
    end
end