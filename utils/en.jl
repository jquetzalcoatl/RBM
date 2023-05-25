using LinearAlgebra, Flux
include("adamOpt.jl")

@doc raw"""
Energy of Binary RBM
```math
E(|v⟩, |h⟩) = - ⟨v|a⟩ - ⟨b|h⟩ - ⟨v|W|h⟩
```
"""
# en(rbm, J) = - (rbm.v' * J.a + J.b' * rbm.h + rbm.v' * J.w * rbm.h)
# en(rbm, J) = mean(- (rbm.v' * J.a + (J.b' * rbm.h)' + [(rbm.v' * J.w * rbm.h)[i,i] for i in 1:hparams.batch_size]))
en(rbm, J) = - mean(rbm.v' * J.a + (J.b' * rbm.h)' + diag(rbm.v' * J.w * rbm.h))
H(rbm, J) = - (rbm.v' * J.a + (J.b' * rbm.h)' + diag(rbm.v' * J.w * rbm.h))
avgEn(rbm,J) = sum(H(rbm, J) .* exp.(-H(rbm, J))) / sum(exp.(- H(rbm,J)))
avgEn2(rbm,J, hparams) = sum(H(rbm, J) .* exp.(-H(rbm, J)/(hparams.nv+hparams.nh))) / sum(exp.(-H(rbm,J)/(hparams.nv+hparams.nh)))


function H_effective(J,hparams)
    F = LinearAlgebra.svd(J.w, full=true);
    Hamiltonian = -( sum([J.a * F.V[:,i]' for i in 1:hparams.nh]) + sum([F.U[:,1] * reshape(J.b,1,:) for i in 1:hparams.nv]) + F.U * dev(cat(Diagonal(F.S), (zeros(size(F.U,1)-size(F.Vt,1),size(F.Vt,1))),dims=1)) * F.Vt)
    return Hamiltonian
end

@doc raw"""
Loss function
```math
Δw\_{ij} = β (⟨vᵢ⋅hⱼ⟩_{p(h),p_{data}} - ⟨vᵢ⋅hⱼ⟩_{p(v,h)})
Δaₗ = β (⟨vₗ⟩_{p(h),p_{data}} - ⟨vₗ⟩_{p(v,h)})
Δbₗ = β (⟨hₗ⟩_{p(h),p_{data}} - ⟨hₗ⟩_{p(v,h)})
```
"""
function loss(rbm, J, x_data, x_Gibbs; hparams, β=1, dev)
    rbm.v = x_data |> dev
    
    rbm.h = Array{Float32}(sign.(rand(hparams.nh, hparams.batch_size) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev
    Z = sum(exp.(H(rbm, J)/(hparams.nv+hparams.nh)))
        
    vh_data = (rbm.v * Diagonal(exp.(H(rbm, J)/(hparams.nv+hparams.nh))) * CuArray(rbm.h')) / Z # / hparams.batch_size 
    v_data = (rbm.v * exp.(H(rbm, J)/(hparams.nv+hparams.nh))) / Z
    h_data = (rbm.h * exp.(H(rbm, J)/(hparams.nv+hparams.nh))) / Z
    for i in 1:2
        rbm.h = Array{Float32}(sign.(rand(hparams.nh, hparams.batch_size) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev
        Z = sum(exp.(H(rbm, J)/(hparams.nv+hparams.nh)))
            
        vh_data_tmp = (rbm.v * Diagonal(exp.(H(rbm, J)/(hparams.nv+hparams.nh))) * CuArray(rbm.h')) / Z
        v_data_tmp = (rbm.v * exp.(H(rbm, J)/(hparams.nv+hparams.nh))) / Z
        h_data_tmp = (rbm.h * exp.(H(rbm, J)/(hparams.nv+hparams.nh))) / Z
    
        vh_data = cat(vh_data, vh_data_tmp, dims=3)
        v_data = cat(v_data, v_data_tmp, dims=3)
        h_data = cat(h_data, h_data_tmp, dims=3)
    end
    vh_data = reshape(mean(vh_data, dims=3), hparams.nv, hparams.nh)
    v_data = reshape(mean(v_data, dims=3), hparams.nv)
    h_data = reshape(mean(h_data, dims=3), hparams.nh)

    rbm.v = x_Gibbs |> dev

    for i in 1:hparams.t
        rbm.h = Array{Float32}(sign.(rand(hparams.nh, hparams.batch_size) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev
        rbm.v = Array{Float32}(sign.(rand(hparams.nv, hparams.batch_size) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
    end

    vh_recontruct = rbm.v * Diagonal(exp.(H(rbm, J)/(hparams.nv+hparams.nh))) * CuArray(rbm.h') / Z #/ hparams.batch_size
    v_reconstruct = rbm.v * exp.(H(rbm, J)/(hparams.nv+hparams.nh)) / Z #/hparams.batch_size
    h_reconstruct = rbm.h * exp.(H(rbm, J)/(hparams.nv+hparams.nh)) / Z #/hparams.batch_size

    Δw = vh_data - vh_recontruct
    Δa = v_data - v_reconstruct
    Δb = h_data - h_reconstruct

    Δw, Δa, Δb
end

function updateJ!(J, Δw, Δa, Δb, opt; hparams)
    if hparams.optType == "Adam"
        updateJAdam!(J, Δw, Δa, Δb, opt; hparams)
    elseif hparams.optType == "SGD"
        updateJSGD!(J, Δw, Δa, Δb; hparams)
    end
end

function updateJSGD!(J, Δw, Δa, Δb; hparams)
    J.w = (1 - hparams.γ) .* J.w + hparams.lr .* Δw
    J.a = J.a + hparams.lr .* Δa
    J.b = J.b + hparams.lr .* Δb
end

function updateJAdam!(J, Δw, Δa, Δb, opt; hparams)
    J.w = step!(opt.w, Δw)
    J.a = step!(opt.a, Δa)
    J.b = step!(opt.b, Δb)
end
    