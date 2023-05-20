using LinearAlgebra, Flux
include("adamOpt.jl")

function genSample(rbm, J, hparams, m; num = 4, t = 10, β = 1, mode = "train", dev)
    
    xh = rand(hparams.nh, num) |> dev
    rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * (rand(hparams.nh, num) |> dev) .+ J.a)))) |> dev

    for i in 1:t
        rbm.h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
    end

    samp = reshape(rbm.v, 28,28,:) |> cpu;

    if mode == "train"
        sampAv = Int(num/4)
        pEn = plot(m.enList, yerr=m.enSDList, label="Energy per spin \n  T = $(round(1/(β+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)

        pLoss = plot(m.ΔwList, yerr=m.ΔwSDList, label="Δw", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pLoss = plot!(m.ΔaList, yerr=m.ΔaSDList, label="Δa", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pLoss = plot!(m.ΔbList, yerr=m.ΔbSDList, label="Δb", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        
        pEigen = computeEigenonW(J, hparams)
        pWMean = plot(m.wMean, yerr=m.wVar, label="w mean", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        

        # avSamp = [σ.(mean(samp[:,:,1 + sampAv*(i-1):sampAv*i], dims=3))[:,:,1] for i in 1:4]
        avSamp = [mean(samp[:,:,1 + sampAv*(i-1):sampAv*i], dims=3)[:,:,1] for i in 1:4]
        hmSamp = heatmap(hcat(avSamp...))
        
        p1 = plot(pEn, pLoss, layout=(1,2))
        p2 = plot(pEigen, pWMean, layout=(1,2))
        p = plot(p1, p2, layout=(2,1))
        f = plot(p,hmSamp, layout=(2,1), size=(500,600))
        display(f)
    elseif mode == "test"
        # avSamp = σ.(mean(samp, dims=3))[:,:,1]
        avSamp = mean(samp, dims=3)[:,:,1]
        hmSamp = heatmap(avSamp)
        display(hmSamp)
    end
end

function computeEigenonW(J, hparams)
    F = LinearAlgebra.svd(J.w)
    lambda = F.S |> cpu
    f = plot(lambda, markershape=:circle, scale=:log10, label="λ Jw", markersize=7, 
    markershapes = :circle, lw=1.5, markerstrokewidth=0, frame=:box)
    W = randn(hparams.nv, hparams.nh) ./ √(2*hparams.nh);
    F = LinearAlgebra.svd(W);
    f = plot!(F.S, markershape=:circle, scale=:log10, label="λ rdm", markersize=7, 
    markershapes = :circle, lw=1.5, markerstrokewidth=0, frame=:box)
    return f
end


function JwByComponent(rbm, J, hparams; num = 4, t = 10, β = 1, idx = 1, dev)
    F = LinearAlgebra.svd(J.w)
    Jw = F.S[idx] * F.U[:,idx] * F.V[:,idx]'
    xh = rand(hparams.nh, num) |> dev
    rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (Jw * (rand(hparams.nh, num) |> dev) .+ J.a)))) |> dev

    for i in 1:10
        rbm.h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (Jw' * rbm.v .+ J.b)))) |> dev 
        rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (Jw * rbm.h .+ J.a)))) |> dev  
    end

    samp = reshape(rbm.v, 28,28,:) |> cpu;

    avSamp = σ.(mean(samp, dims=3))[:,:,1]
    hmSamp = heatmap(avSamp, title=F.S[idx])
    display(hmSamp)
end

function MatrixMean(m)
    w = m * m' ./ size(m,2)
    matMean = tr(w)/size(w,1)
    return matMean
end

function MatrixVar(m)
    w = m * m' ./ size(m,2)
    matVar = tr(w)/size(w,1)
    return matVar
end