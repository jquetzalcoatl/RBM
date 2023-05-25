using LinearAlgebra, Flux, CUDA
include("adamOpt.jl")
include("en.jl")

function genSample(rbm, J, hparams, m; num = 25, t = 10, β = 1, mode = "train", plotSample=true, epoch=0, dict, dev)
    lnum = Int(sqrt(num))
    xh = rand(hparams.nh, num) |> dev
    rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * (rand(hparams.nh, num) |> dev) .+ J.a)))) |> dev

    for i in 1:t
        rbm.h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
    end

    samp = reshape(rbm.v, 28,28,:) |> cpu;

    if mode == "train"
        # sampAv = Int(num/4)
        pEn = plot(.- m.enList .- log.(m.Z), yerr=m.enSDList, label="e T=$(round(1/(β+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pEn = plot!(.- m.enZList .- log.(m.Z), markersize=7, markershapes = :circle, lw=1.5)

        pLoss = plot(m.ΔwList, yerr=m.ΔwSDList, label="Δw", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pLoss = plot!(m.ΔaList, yerr=m.ΔaSDList, label="Δa", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pLoss = plot!(m.ΔbList, yerr=m.ΔbSDList, label="Δb", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        
        pEigen = computeEigenonW(J, hparams)
        pWMean = plot(m.wMean, yerr=m.wVar, label="w mean", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pWMean = plot!(m.wTrMean, yerr=m.wTrVar, label="w mean", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        

        # avSamp = [σ.(mean(samp[:,:,1 + sampAv*(i-1):sampAv*i], dims=3))[:,:,1] for i in 1:4]
        # avSamp = [mean(samp[:,:,1 + sampAv*(i-1):sampAv*i], dims=3)[:,:,1] for i in 1:4]
        # hmSamp = heatmap(hcat(avSamp...))
        avSamp = cat([cat([samp[:,:,i+j*lnum] for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
        hmSamp = heatmap(permutedims(avSamp,(2,1)))
        
        p1 = plot(pEn, pLoss, layout=(1,2))
        p2 = plot(pEigen, pWMean, layout=(1,2))
        p = plot(p1, p2, layout=(2,1))
        f = plot(p,hmSamp, layout=(2,1), size=(500,600))
        if plotSample
            display(f)
        else
            isdir(dict["bdir"] * "/models/$(dict["msg"])/Plots") || mkpath(dict["bdir"] * "/models/$(dict["msg"])/Plots")
            savefig(f, dict["bdir"] * "/models/$(dict["msg"])/Plots/$epoch.png")
        end
    elseif mode == "test"
        # avSamp = σ.(mean(samp, dims=3))[:,:,1]
        # avSamp = mean(samp, dims=3)[:,:,1]
        avSamp = cat([cat([samp[:,:,i+j*lnum] for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
        hmSamp = heatmap(permutedims(avSamp,(2,1)))
        display(hmSamp)
    elseif mode == "results"
        # pEn = plot(m.enList, yerr=m.enSDList, label="e T=$(round(1/(β+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pEn = plot(.- m.enList .- log.(m.Z), ribbon=m.enSDList, lw=1.5, label="e T=$(round(1/(β+0.000001), digits=2))")
        pEn = plot!(.- m.enZList .- log.(m.Z), lw=1.5)

        pLoss = plot(m.ΔwList, ribbon=m.ΔwSDList, label="Δw", lw=1.5)
        pLoss = plot!(m.ΔaList, ribbon=m.ΔaSDList, label="Δa", lw=1.5)
        pLoss = plot!(m.ΔbList, ribbon=m.ΔbSDList, label="Δb", lw=1.5)
        
        pEigen = computeEigenonW(J, hparams)
        pWMean = plot(m.wMean, ribbon=m.wVar, label="w mean", lw=1.5)
        pWMean = plot!(m.wTrMean, ribbon=m.wTrVar, label="w mean", lw=1.5)
        
        p1 = plot(pEn, pLoss, layout=(1,2))
        p2 = plot(pEigen, pWMean, layout=(1,2))
        p = plot(p1, p2, layout=(2,1), size=(500,600))
        display(p)
    end
end

function computeEigenonW(J, hparams)
    F = LinearAlgebra.svd(J.w)
    lambda = F.S |> cpu
    f = plot(lambda, markershape=:circle, scale=:log10, label="λ Jw", markersize=7, 
    markershapes = :circle, lw=1.5, markerstrokewidth=0, frame=:box)
    W = randn(hparams.nv, hparams.nh) .* 0.1 / √(hparams.nh);
    F = LinearAlgebra.svd(W);
    f = plot!(F.S, markershape=:circle, scale=:log10, label="λ rdm", markersize=7, 
    markershapes = :circle, lw=1.5, markerstrokewidth=0, frame=:box)
    return f
end


function JwByComponent(rbm, J, hparams; num = 4, t = 10, β = 1, idx = 1, dev)
    # Heff = H_effective(J,hparams) #ToAdd
    F = LinearAlgebra.svd(J.w, full=true)
    Jw = F.S[idx] * F.U[:,idx] * F.V[:,idx]'    # I could also have used F.Vt instead of F.V'
    xh = rand(hparams.nh, num) |> dev
    rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (Jw * (rand(hparams.nh, num) |> dev) .+ J.a)))) |> dev

    for i in 1:10
        rbm.h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (Jw' * rbm.v .+ J.b)))) |> dev 
        rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (Jw * rbm.h .+ J.a)))) |> dev  
    end

    samp = reshape(rbm.v, 28,28,:) |> cpu;

    # avSamp = σ.(mean(samp, dims=3))[:,:,1]
    avSamp = mean(samp, dims=3)[:,:,1]
    hmSamp = heatmap(avSamp, title=F.S[idx])
    display(hmSamp)
end

# CUDA.@allowscalar
function MatrixMean(m)
    # # CUDA.allowscalar() do 
    # mN = (m .- mean(m, dims=1)) ./ std(m, dims=1)
    # # end
    # w = mN * mN' ./ size(mN,2)
    w = m * m' ./ size(m,2)
    matMean = tr(w)/size(w,1)
    return matMean
end

# CUDA.@allowscalar
function MatrixVar(m)
    # # CUDA.allowscalar() do 
    # mN = (m .- mean(m, dims=1)) ./ std(m, dims=1)
    # # end
    # w = mN * mN' ./ size(mN,2)
    w = m * m' ./ size(m,2)
    matVar = tr(w*w')/size(w,1)
    return matVar
end

function genEnZSample(rbmZ, J, hparams, m; sampleSize = 1000, t_samp = 10, β = 1, dev)
    
    xh = rand(hparams.nh, sampleSize) |> dev
    rbmZ.v = Array{Float32}(sign.(rand(hparams.nv, sampleSize) |> dev .< σ.(β .* (J.w * (rand(hparams.nh, sampleSize) |> dev) .+ J.a)))) |> dev

    for i in 1:t_samp
        rbmZ.h = Array{Float32}(sign.(rand(hparams.nh, sampleSize) |> dev .< σ.(β .* (J.w' * rbmZ.v .+ J.b)))) |> dev 
        rbmZ.v = Array{Float32}(sign.(rand(hparams.nv, sampleSize) |> dev .< σ.(β .* (J.w * rbmZ.h .+ J.a)))) |> dev  
    end
    avgEn2(rbmZ,J, hparams)
end

function amplitudes(J, hparams; num=1, β=1.0, t_samp=40, ϵ=1e-5, dev=gpu )
    num=1
    F = LinearAlgebra.svd(J.w, full=true)
    batchV = zeros(hparams.nv, t_samp)
    batchH = zeros(hparams.nh, t_samp)
    
    Xv = CuArray{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * (rand(hparams.nh, num) |> dev) .+ J.a)))) |> dev
    a = abs.(F.U' * Xv)
    # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
    # batchV[:,1] =  reshape(c,:);
    batchV[:,1] =  reshape(a,:);
    
    
    for i in 2:t_samp
        Xh = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * Xv .+ J.b)))) |> dev 
        a = abs.(F.V' * Xh)
        # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
        # batchH[:,i] =  reshape(c,:)
        batchH[:,i] =  reshape(a,:)
        
        Xv = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * Xh .+ J.a)))) |> dev 
        a = abs.(F.U' * Xv)
        # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
        # batchV[:,i] = reshape(c,:);
        batchV[:,i] = reshape(a,:);
    end
    return batchV, batchH[:,2:end]
end

function amplitudes2(J, hparams; num=1, β=1.0, t_samp=40, ϵ=1e-5, dev=gpu )
    num=1
    Heff = H_effective(J,hparams)
    # F = LinearAlgebra.svd(J.w, full=true)
    F = LinearAlgebra.svd(Heff, full=true)
    batchV = zeros(hparams.nv, t_samp)
    batchH = zeros(hparams.nh, t_samp)
    
    Xv = CuArray{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * (rand(hparams.nh, num) |> dev) .+ J.a)))) |> dev
    a = abs.(F.U' * Xv)
    batchV[:,1] =  reshape(a,:);
    
    
    for i in 2:t_samp
        Xh = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * Xv .+ J.b)))) |> dev 
        a = abs.(F.V' * Xh)
        # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
        # batchH[:,i] =  reshape(c,:)
        batchH[:,i] =  reshape(a,:)
        
        Xv = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * Xh .+ J.a)))) |> dev 
        a = abs.(F.U' * Xv)
        # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
        # batchV[:,i] = reshape(c,:);
        batchV[:,i] = reshape(a,:);
    end
    return batchV, batchH[:,2:end]
end

function correlation(rbm, J, hparams; t_therm=10000, t_corr=10000, β=1)
    corr = []
    xh = rand(hparams.nh) |> dev
    rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * xh .+ J.a)))) |> dev

    for i in 1:t_therm
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
    end

    rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
    m = σ.(β .* (J.w * rbm.h .+ J.a))
    rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< m)) |> dev
    
    Δv0 = m .- mean(m, dims=1)
    Δv = Δv0
    C0 = (Δv0' * Δv0)/hparams.nv
    Ct = (Δv0' * Δv)/hparams.nv
    append!(corr, Ct/C0)
    for i in 1:t_corr
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        m = σ.(β .* (J.w * rbm.h .+ J.a))
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< m)) |> dev
        Δv = m .- mean(m, dims=1)
        Ct = (Δv0' * Δv)/hparams.nv
        
        append!(corr, Ct/C0)
    end
    corr
end

function correlation2(rbm, J, hparams; t_therm=10000, t_corr=10000, β=1)
    corr = []
    xh = rand(hparams.nh) |> dev
    rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * xh .+ J.a)))) |> dev

    for i in 1:t_therm
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
    end

    rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
    rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
    m = rbm.v
    
    Δv0 = m .- mean(m, dims=1)
    Δv = Δv0
    C0 = (Δv0' * Δv0)/hparams.nv
    Ct = (Δv0' * Δv)/hparams.nv
    append!(corr, Ct/C0)
    for i in 1:t_corr
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
        m = rbm.v
        Δv = m .- mean(m, dims=1)
        Ct = (Δv0' * Δv)/hparams.nv
        
        append!(corr, Ct/C0)
    end
    corr
end




