using LinearAlgebra, Flux

@doc raw"""
Energy of Binary RBM
```math
E(|v⟩, |h⟩) = - ⟨v|a⟩ - ⟨b|h⟩ - ⟨v|W|a⟩
```
"""
en(rbm, J) = - (rbm.v' * J.a + J.b' * rbm.h + rbm.v' * J.w * rbm.h)
en(rbm, J) = mean(- (rbm.v' * J.a + (J.b' * rbm.h)' + [(rbm.v' * J.w * rbm.h)[i,i] for i in 1:hparams.batch_size]))
en(rbm, J) = - mean(rbm.v' * J.a + (J.b' * rbm.h)' + diag(rbm.v' * J.w * rbm.h))

@doc raw"""
Loss function
```math
Δw\_{ij} = β (⟨vᵢ⋅hⱼ⟩_{p(h),p_{data}} - ⟨vᵢ⋅hⱼ⟩_{p(v,h)})
Δaₗ = β (⟨vₗ⟩_{p(h),p_{data}} - ⟨vₗ⟩_{p(v,h)})
Δbₗ = β (⟨hₗ⟩_{p(h),p_{data}} - ⟨hₗ⟩_{p(v,h)})
```
"""
function loss(rbm, J, x; hparams, β=1)
    rbm.v = x
    rbm.h = Array{Float32}(sign.(rand(hparams.nh) .< σ.(β .* (J.w' * rbm.v .+ J.b)))) 

    vh_data = (rbm.v * rbm.h')/hparams.batch_size
    v_data = reshape(mean(rbm.v, dims=2),:)/hparams.batch_size
    h_data = reshape(mean(rbm.h, dims=2),:)/hparams.batch_size

    rbm.v = Array{Float32}(sign.(rand(hparams.nv) .< σ.(β .* (J.w * rbm.h .+ J.a))))  

    for i in 1:hparams.t
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) .< σ.(β .* (J.w' * rbm.v .+ J.b)))) 
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) .< σ.(β .* (J.w * rbm.h .+ J.a))))  
    end

    vh_recontruct = (rbm.v * rbm.h')/hparams.batch_size
    v_reconstruct = reshape(mean(rbm.v, dims=2),:)/hparams.batch_size
    h_reconstruct = reshape(mean(rbm.h, dims=2),:)/hparams.batch_size

    Δw = vh_data - vh_recontruct
    Δa = v_data - v_reconstruct
    Δb = h_data - h_reconstruct

    Δw, Δa, Δb
end

function updateJ!(J, Δw, Δa, Δb; hparams)
    J.w = J.w + hparams.lr .* Δw
    J.a = J.a + hparams.lr .* Δa
    J.b = J.b + hparams.lr .* Δb
end

function genSample(rbm, J, hparams, m; num = 4, t = 10, β = 1, mode = "train")
    xh = rand(hparams.nh, num)
    rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) .< σ.(β .* (J.w * rand(hparams.nh, num) .+ J.a))))

    for i in 1:t
        rbm.h = Array{Float32}(sign.(rand(hparams.nh, num) .< σ.(β .* (J.w' * rbm.v .+ J.b)))) 
        rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) .< σ.(β .* (J.w * rbm.h .+ J.a))))  
    end

    samp = reshape(rbm.v, 28,28,:);

    if mode == "train"
        pEn = plot(m.enList, label="Energy", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0)
        pLoss = plot(m.ΔwList, label="Loss w", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0)
        pLoss = plot!(m.ΔaList, label="Loss a", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0)
        pLoss = plot!(m.ΔbList, label="Loss b", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0)
        hmSamp = heatmap(reshape(samp[:,:,1:4], 28,28*4))
        p = plot(pEn, pLoss, layout=(1,2))
        f = plot(p,hmSamp, layout=(2,1))
        display(f)
    elseif mode == "test"
        avSamp = mean(samp, dims=3)
        hmSamp = heatmap(avSamp)
        display(hmSamp)
    end
end