using LinearAlgebra, Flux
include("adamOpt.jl")

@doc raw"""
Energy of Binary RBM
```math
E(|v⟩, |h⟩) = - ⟨v|a⟩ - ⟨b|h⟩ - ⟨v|W|a⟩
```
"""
# en(rbm, J) = - (rbm.v' * J.a + J.b' * rbm.h + rbm.v' * J.w * rbm.h)
# en(rbm, J) = mean(- (rbm.v' * J.a + (J.b' * rbm.h)' + [(rbm.v' * J.w * rbm.h)[i,i] for i in 1:hparams.batch_size]))
en(rbm, J) = - mean(rbm.v' * J.a + (J.b' * rbm.h)' + diag(rbm.v' * J.w * rbm.h))
H(rbm, J) = - (rbm.v' * J.a + (J.b' * rbm.h)' + diag(rbm.v' * J.w * rbm.h))
avgEn(rbm,J) = sum(H(rbm, J) .* exp.(H(rbm, J))) / sum(exp.(H(rbm,J)))
@doc raw"""
Loss function
```math
Δw\_{ij} = β (⟨vᵢ⋅hⱼ⟩_{p(h),p_{data}} - ⟨vᵢ⋅hⱼ⟩_{p(v,h)})
Δaₗ = β (⟨vₗ⟩_{p(h),p_{data}} - ⟨vₗ⟩_{p(v,h)})
Δbₗ = β (⟨hₗ⟩_{p(h),p_{data}} - ⟨hₗ⟩_{p(v,h)})
```
"""
function loss(rbm, J, x; hparams, β=1, dev)
    rbm.v = x |> dev
    rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev

    vh_data = (rbm.v * rbm.h')/hparams.batch_size
    v_data = reshape(mean(rbm.v, dims=2),:)/hparams.batch_size
    h_data = reshape(mean(rbm.h, dims=2),:)/hparams.batch_size

    rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev

    for i in 1:hparams.t
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
    end

    vh_recontruct = (rbm.v * rbm.h')/hparams.batch_size
    v_reconstruct = reshape(mean(rbm.v, dims=2),:)/hparams.batch_size
    h_reconstruct = reshape(mean(rbm.h, dims=2),:)/hparams.batch_size

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
    J.w = J.w + hparams.lr .* Δw
    J.a = J.a + hparams.lr .* Δa
    J.b = J.b + hparams.lr .* Δb
end

function updateJAdam!(J, Δw, Δa, Δb, opt; hparams)
    J.w = step!(opt.w, Δw)
    J.a = step!(opt.a, Δa)
    J.b = step!(opt.b, Δb)
end

# function genSample(rbm, J, hparams, m; num = 4, t = 10, β = 1, mode = "train", dev)
    
#     xh = rand(hparams.nh, num) |> dev
#     rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * (rand(hparams.nh, num) |> dev) .+ J.a)))) |> dev

#     for i in 1:t
#         rbm.h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
#         rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
#     end

#     samp = reshape(rbm.v, 28,28,:) |> cpu;

#     if mode == "train"
#         sampAv = Int(num/4)
#         pEn = plot(m.enList, yerr=m.enSDList, label="Energy per spin \n  T = $(round(1/(β+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0)

#         pLoss = plot(m.ΔwList, yerr=m.ΔwSDList, label="Δw", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0)
#         pLoss = plot!(m.ΔaList, yerr=m.ΔaSDList, label="Δa", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0)
#         pLoss = plot!(m.ΔbList, yerr=m.ΔbSDList, label="Δb", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0)
        
#         pEigen = computeEigenonW(J, hparams)
#         pWMean = plot(m.wMean, label="w mean", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0)
        

#         avSamp = [σ.(mean(samp[:,:,1 + sampAv*(i-1):sampAv*i], dims=3))[:,:,1] for i in 1:4]
#         hmSamp = heatmap(hcat(avSamp...))
        
#         p1 = plot(pEn, pLoss, layout=(1,2))
#         p2 = plot(pEigen, pWMean, layout=(1,2))
#         p = plot(p1, p2, layout=(2,1))
#         f = plot(p,hmSamp, layout=(2,1), size=(500,600))
#         display(f)
#     elseif mode == "test"
#         avSamp = σ.(mean(samp, dims=3))[:,:,1]
#         hmSamp = heatmap(avSamp)
#         display(hmSamp)
#     end
# end

# function computeEigenonW(J, hparams)
#     F = LinearAlgebra.svd(J.w)
#     lambda = F.S |> cpu
#     f = plot(lambda, markershape=:circle, scale=:log10, label="λ Jw", markersize=7, 
#     markershapes = :circle, lw=1.5, markerstrokewidth=0, frame=:box)
#     W = randn(hparams.nv, hparams.nh) ./ √(2*hparams.nh);
#     F = LinearAlgebra.svd(W);
#     f = plot!(F.S, markershape=:circle, scale=:log10, label="λ rdm", markersize=7, 
#     markershapes = :circle, lw=1.5, markerstrokewidth=0, frame=:box)
#     return f
# end


# function JwByComponent(rbm, J, hparams; num = 4, t = 10, β = 1, idx = 1, dev)
#     F = LinearAlgebra.svd(J.w)
#     Jw = F.S[idx] * F.U[:,idx] * F.V[:,idx]'
#     xh = rand(hparams.nh, num) |> dev
#     rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (Jw * (rand(hparams.nh, num) |> dev) .+ J.a)))) |> dev

#     for i in 1:10
#         rbm.h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (Jw' * rbm.v .+ J.b)))) |> dev 
#         rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (Jw * rbm.h .+ J.a)))) |> dev  
#     end

#     samp = reshape(rbm.v, 28,28,:) |> cpu;

#     avSamp = σ.(mean(samp, dims=3))[:,:,1]
#     hmSamp = heatmap(avSamp, title=F.S[idx])
#     display(hmSamp)
# end

# function computeMatrixMean(J)
#     w = tr(J.w * J.w')
#     matMean = w/size(J.w,1)
#     return matMean
# end
    