using LinearAlgebra, Flux
include("adamOpt.jl")

@doc raw"""
Energy of Binary RBM
```math
E(|v⟩, |h⟩) = - ⟨v|a⟩ - ⟨b|h⟩ - ⟨v|W|h⟩
```
"""
en(rbm, J) = - mean(rbm.v' * J.a + (J.b' * rbm.h)' + diag(rbm.v' * J.w * rbm.h))
H(rbm, J) = - (rbm.v' * J.a + (J.b' * rbm.h)' + diag(rbm.v' * J.w * rbm.h))
avgEn(rbm,J, β) = sum(H(rbm, J) .* exp.(-β .* H(rbm, J))) / sum(exp.(- β .* H(rbm,J)))
# avgEn2(rbm,J, hparams) = sum(H(rbm, J) .* exp.(-H(rbm, J)/(hparams.nv+hparams.nh))) / sum(exp.(-H(rbm,J)/(hparams.nv+hparams.nh)))


function H_effective(J,hparams; dev)
    F = LinearAlgebra.svd(J.w, full=true);
    if !isapprox(F.U * dev(cat(Diagonal(F.S), (zeros(hparams.nv-hparams.nh,hparams.nh)),dims=1)) * F.Vt, J.w, atol=1e-2)
        @warn "Diagonalization of J.w prompted false"
    end
    
    Hamiltonian = -( repeat(J.a, 1,hparams.nh) * F.Vt + 
        F.U * repeat(reshape(J.b,1,:), hparams.nv,1) + 
        F.U * dev(cat(Diagonal(F.S), (zeros(hparams.nv-hparams.nh,hparams.nh)),dims=1)) * F.Vt)
    
    F = LinearAlgebra.svd( Hamiltonian, full=true)
    if !isapprox(F.U * dev(cat(Diagonal(F.S), (zeros(hparams.nv-hparams.nh,hparams.nh)),dims=1)) * F.Vt, Hamiltonian, atol=1e-2)
        @warn "Diagonalization of effective H prompted false"
    end
    if !isapprox(F.U * F.U', I)
        @warn "U eigenvectors don't span space"
        exit()
    end
    if !isapprox(F.Vt * F.V, I)
        @warn "V eigenvectors don't span space"
    end
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
function loss(rbm, J, x_data, x_Gibbs; hparams, β=1, β2=1, dev, lProtocol="Rdm", bw=true, N=4)
    
    rbm.v = x_data |> dev
    v_data = rbm.v
    h_data = exp.(β2 * (J.b .+ J.w' * rbm.v)) ./ (1 .+ exp.(β2 * (J.b .+ J.w' * rbm.v)))
    vh_data = (v_data * h_data')/hparams.batch_size
    v_data = reshape(mean(v_data,dims=2),:)
    h_data = reshape(mean(h_data,dims=2),:)

    rbm.v = x_Gibbs |> dev

    if lProtocol in ["Rdm", "CD", "PCD"]
        for i in 1:hparams.t
            rbm.h = Array{Float32}(sign.(rand(hparams.nh, hparams.batch_size) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev
            rbm.v = Array{Float32}(sign.(rand(hparams.nv, hparams.batch_size) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev 
        end
        if bw
            Z = sum(exp.(- β2 .* H(rbm, J)))
            vh_recontruct = rbm.v * Diagonal(exp.(- β2 .* H(rbm, J))) * CuArray(rbm.h') / Z
            v_reconstruct = rbm.v * exp.(- β2 .* H(rbm, J)) / Z
            h_reconstruct = rbm.h * exp.(- β2 .* H(rbm, J)) / Z
        else
            vh_recontruct = rbm.v * CuArray(rbm.h')
            v_reconstruct = reshape(mean(rbm.v, dims=2),:)
            h_reconstruct = reshape(mean(rbm.h, dims=2),:)
        end
    end
    Δw = vh_data - vh_recontruct #- hparams.γ .* J.w
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
    J.w = J.w + hparams.lr .* Δw
    J.a = J.a + hparams.lr .* Δa
    J.b = J.b + hparams.lr .* Δb
end

function updateJAdam!(J, Δw, Δa, Δb, opt; hparams)
    J.w = step!(opt.w, Δw) #(1 - hparams.γ) .* 
    J.a = step!(opt.a, Δa)
    J.b = step!(opt.b, Δb)
end
    
##################
# function loss(rbm, J, x_data, x_Gibbs; hparams, β=1, β2=1, dev, lProtocol="Rdm", bw=true, N=4)
    
#     rbm.v = x_data |> dev
    
#     rbm.h = Array{Float32}(sign.(rand(hparams.nh, hparams.batch_size) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev
#     if bw
#         Z = sum(exp.(- β2 .* H(rbm, J)))
#         vh_data = (rbm.v * Diagonal(exp.(- β2 .* H(rbm, J))) * CuArray(rbm.h')) / Z
#         v_data = (rbm.v * exp.(- β2 .* H(rbm, J))) / Z
#         h_data = (rbm.h * exp.(- β2 .* H(rbm, J))) / Z
#         for i in 1:N
#             rbm.v = x_data |> dev
#             rbm.h = Array{Float32}(sign.(rand(hparams.nh, hparams.batch_size) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev

#             Z = sum(exp.(- β2 .* H(rbm, J)))

#             vh_data_tmp = (rbm.v * Diagonal(exp.(- β2 .* H(rbm, J))) * CuArray(rbm.h')) / Z
#             v_data_tmp = (rbm.v * exp.(- β2 .* H(rbm, J))) / Z
#             h_data_tmp = (rbm.h * exp.(- β2 .* H(rbm, J))) / Z

#             vh_data = cat(vh_data, vh_data_tmp, dims=3)
#             v_data = cat(v_data, v_data_tmp, dims=3)
#             h_data = cat(h_data, h_data_tmp, dims=3)
#         end
#     else
#         vh_data = rbm.v * CuArray(rbm.h')
#         v_data = reshape(mean(rbm.v,dims=2),:)
#         h_data = reshape(mean(rbm.h,dims=2),:)
#         for i in 1:N
#             rbm.v = x_data |> dev
#             rbm.h = Array{Float32}(sign.(rand(hparams.nh, hparams.batch_size) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev

#             vh_data_tmp = rbm.v * CuArray(rbm.h')
#             v_data_tmp = reshape(mean(rbm.v,dims=2),:)
#             h_data_tmp = reshape(mean(rbm.h,dims=2),:)

#             vh_data = cat(vh_data, vh_data_tmp, dims=3)
#             v_data = cat(v_data, v_data_tmp, dims=3)
#             h_data = cat(h_data, h_data_tmp, dims=3)
#         end
#     end
#     vh_data = reshape(mean(vh_data, dims=3), hparams.nv, hparams.nh)
#     v_data = reshape(mean(v_data, dims=3), hparams.nv)
#     h_data = reshape(mean(h_data, dims=3), hparams.nh)

#     rbm.v = x_Gibbs |> dev

#     if lProtocol in ["Rdm", "CD", "PCD"]
#         for i in 1:hparams.t
#             rbm.h = Array{Float32}(sign.(rand(hparams.nh, hparams.batch_size) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev
#             rbm.v = Array{Float32}(sign.(rand(hparams.nv, hparams.batch_size) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev 
#         end
#         if bw
#             Z = sum(exp.(- β2 .* H(rbm, J)))
#             vh_recontruct = rbm.v * Diagonal(exp.(- β2 .* H(rbm, J))) * CuArray(rbm.h') / Z
#             v_reconstruct = rbm.v * exp.(- β2 .* H(rbm, J)) / Z
#             h_reconstruct = rbm.h * exp.(- β2 .* H(rbm, J)) / Z
#         else
#             vh_recontruct = rbm.v * CuArray(rbm.h')
#             v_reconstruct = reshape(mean(rbm.v, dims=2),:)
#             h_reconstruct = reshape(mean(rbm.h, dims=2),:)
#         end
        
#         Δw = vh_data - vh_recontruct - hparams.γ .* J.w
#         Δa = v_data - v_reconstruct
#         Δb = h_data - h_reconstruct
#     end

#     Δw, Δa, Δb
# end
