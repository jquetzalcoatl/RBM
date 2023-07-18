using LinearAlgebra, Flux, CUDA
include("adamOpt.jl")
include("en.jl")

function genSample(rbm, J, hparams, m; num = 25, t = 10, β = 1, β2=1, mode = "train", plotSample=true, epoch=0, dict, dev, TS)
    lnum = Int(sqrt(num))
    xh = sign.(rand(hparams.nh, num) .< 0.5) |> dev
    # xh = xh ./ sum(xh, dims=1)
    rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * xh .+ J.a)))) |> dev
    # rbm.v = rbm.v ./ sum(rbm.v, dims=1)

    for i in 1:t
        rbm.h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
        rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
        # rbm.v = rbm.v ./ sum(rbm.v, dims=1)
    end

    samp = reshape(rbm.v, 28,28,:) |> cpu;

    if mode == "train"
        pF = plot(.- m.T .* log.(m.Zdata), label="F_d T=$(round(1/(β2+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pF = plot!(.- m.T .* log.(m.Zrbm), markersize=7, markershapes = :circle, lw=1.5, label="F_r")

        pEn = plot(m.enData, label="E_d T=$(round(1/(β2+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pEn = plot!(m.enRBM, markersize=7, markershapes = :circle, lw=1.5, label="E_r")

        pEnt = plot(m.enData ./  m.T .+ log.(m.Zdata), label="S_d T=$(round(1/(β2+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pEnt = plot!(m.enRBM ./ m.T .+ log.(m.Zrbm), markersize=7, markershapes = :circle, lw=1.5, label="S_r")

        pLoss = plot(m.ΔwList, yerr=m.ΔwSDList, label="Δw", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pLoss = plot!(m.ΔaList, yerr=m.ΔaSDList, label="Δa", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pLoss = plot!(m.ΔbList, yerr=m.ΔbSDList, label="Δb", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        
        pEigen = saddlePointEnergySpectrum(J, hparams; dev)
        pWMean = plot(m.wMean, yerr=m.wVar, label="w mean", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pWMean = plot!(m.wTrMean, yerr=m.wTrVar, label="w mean", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        
        avSamp = cat([cat([samp[:,:,i+j*lnum] for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
        hmSamp = heatmap(avSamp, rotate=90)

        pLS = plotLandscapes(rbm, J, lnum ; τ=t, TS, dev, hparams)
        
        p1 = plot(pEn, pLoss, layout=(1,2))
        p2 = plot(pF, pEnt, layout=(1,2))
        p3 = plot(pEigen, pWMean, layout=(1,2))
        p = plot(p1, p2, p3, layout=(3,1))
        f = plot(p,hmSamp, layout=(2,1), size=(900,1200), margin = 8*Plots.mm)
        # f = plot(f, pLS, layout=(2,1), size=(900,1500), margin = 8*Plots.mm)
        if plotSample
            display(f)
            display(pLS)
        else
            isdir(dict["bdir"] * "/models/$(dict["msg"])/Plots") || mkpath(dict["bdir"] * "/models/$(dict["msg"])/Plots")
            savefig(f, dict["bdir"] * "/models/$(dict["msg"])/Plots/$epoch.png")
            savefig(pLS, dict["bdir"] * "/models/$(dict["msg"])/Plots/$(epoch)_LS.png")
        end
    elseif mode == "test"
        avSamp = cat([cat([samp[:,:,i+j*lnum] for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
        hmSamp = heatmap(avSamp, rotate=90)
        display(hmSamp)
    elseif mode == "results"
        pF = plot(.- m.T .* log.(m.Zdata), label="F_d T=$(round(1/(β2+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pF = plot!(.- m.T .* log.(m.Zrbm), markersize=7, markershapes = :circle, lw=1.5, label="F_r")

        pEn = plot(m.enData, label="E_d T=$(round(1/(β2+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pEn = plot!(m.enRBM, markersize=7, markershapes = :circle, lw=1.5, label="E_r")

        pEnt = plot(m.enData ./  m.T .+ log.(m.Zdata), label="S_d T=$(round(1/(β2+0.000001), digits=2))", markersize=7, markershapes = :circle, lw=1.5, markerstrokewidth=0.5)
        pEnt = plot!(m.enRBM ./ m.T .+ log.(m.Zrbm), markersize=7, markershapes = :circle, lw=1.5, label="S_r")

        pLoss = plot(m.ΔwList, ribbon=m.ΔwSDList, label="Δw", lw=1.5)
        pLoss = plot!(m.ΔaList, ribbon=m.ΔaSDList, label="Δa", lw=1.5)
        pLoss = plot!(m.ΔbList, ribbon=m.ΔbSDList, label="Δb", lw=1.5)
        
        pEigen = saddlePointEnergySpectrum(J, hparams; dev)
        pWMean = plot(m.wMean, ribbon=m.wVar, label="w mean", lw=1.5)
        pWMean = plot!(m.wTrMean, ribbon=m.wTrVar, label="w mean", lw=1.5)
        
        p1 = plot(pEn, pLoss, layout=(1,2))
        p2 = plot(pF, pEnt, layout=(1,2))
        p3 = plot(pEigen, pWMean, layout=(1,2))
        p = plot(p1, p2, p3, layout=(3,1), size=(600,800), margin = 6*Plots.mm)
        display(p)
    end
end

function computeEigenonW(J, hparams; dev)
    H_eff = H_effective(J,hparams; dev)
    F = LinearAlgebra.svd(H_eff)
    lambda = F.S |> cpu
    f = plot(J.a*J.b/(lambda .+ 1.e-10), markershape=:circle, label="λ Jw", markersize=7, 
    markershapes = :circle, lw=1.5, markerstrokewidth=0, frame=:box)
    W = randn(hparams.nv, hparams.nh) .* 0.1 / √(hparams.nh);
    F = LinearAlgebra.svd(W);
    f = plot!(F.S, markershape=:circle, label="λ rdm", markersize=7, 
    markershapes = :circle, lw=1.5, scale=:log10, markerstrokewidth=0, frame=:box)
    return f
end

function saddlePointEnergySpectrum(J, hparams; dev)
    F = LinearAlgebra.svd(J.w, full=false);
    a0 = transpose(F.U) * J.a |> cpu
    b0 = F.Vt * J.b  |> cpu ;
    λ = F.S |> cpu ;
    sp_energy = sort(a0 .* b0 ./ (λ .+ 1.e-10))
    f = plot(sp_energy,  markershape=:circle, label=sum(sp_energy), markersize=7, 
    markershapes = :circle, lw=1.5, markerstrokewidth=0, frame=:box)
    return f
end


function JwByComponent(rbm, J, hparams; num = 4, t = 10, β = 1, idx = 1, dev)
    # Heff = H_effective(J,hparams) #ToAdd
    F = LinearAlgebra.svd(J.w, full=true)
    Jw = F.S[idx] * F.U[:,idx] * F.V[:,idx]'    # I could also have used F.Vt instead of F.V'
    xh = sign.(rand(hparams.nh, num) .< 0.5) |> dev
    # xh = xh ./ sum(xh, dims=1)
    rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (Jw * xh .+ J.a)))) |> dev
    # rbm.v = rbm.v ./ sum(rbm.v, dims=1)

    for i in 1:10
        rbm.h = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (Jw' * rbm.v .+ J.b)))) |> dev 
        # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
        rbm.v = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (Jw * rbm.h .+ J.a)))) |> dev  
        # rbm.v = rbm.h ./ sum(rbm.v, dims=1)
    end

    samp = reshape(rbm.v, 28,28,:) |> cpu;

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

function EnData(rbmZ, J, hparams; sampleSize = 1000, t_samp = 10, β = 1, β2=1, dev)
    xh = sign.(rand(hparams.nh, sampleSize) .< 0.5) |> dev
    # xh = xh ./ sum(xh, dims=1)
    rbmZ.v = Array{Float32}(sign.(rand(hparams.nv, sampleSize) |> dev .< σ.(β .* (J.w * xh .+ J.a)))) |> dev
    # rbm.v = rbm.v ./ sum(rbm.v, dims=1)

    for i in 1:t_samp
        rbmZ.h = Array{Float32}(sign.(rand(hparams.nh, sampleSize) |> dev .< σ.(β .* (J.w' * rbmZ.v .+ J.b)))) |> dev 
        # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
        rbmZ.v = Array{Float32}(sign.(rand(hparams.nv, sampleSize) |> dev .< σ.(β .* (J.w * rbmZ.h .+ J.a)))) |> dev  
        # rbm.v = rbm.v ./ sum(rbm.v, dims=1)
    end
    avgEn(rbmZ,J, β2),  sum(exp.(-β2 .* H(rbm,J)))
end

function EnRBM(J, hparams, β2=1; dev)    
    H_eff = H_effective(J,hparams; dev)
    F = LinearAlgebra.svd(H_eff, full=true);
    Z = sum(exp.(-β2 .* F.S))
    sum( F.S' * exp.(-β2 .* F.S))/Z, Z
end

function amplitudes(J, hparams; num=1, β=1.0, t_samp=40, ϵ=1e-5, dev=gpu )
    num=1
    F = LinearAlgebra.svd(J.w, full=true)
    batchV = zeros(hparams.nv, t_samp)
    batchH = zeros(hparams.nh, t_samp)

    xh = sign.(rand(hparams.nh, num) .< 0.5) |> dev
    # xh = xh ./ sum(xh, dims=1)
    Xv = CuArray{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * xh .+ J.a)))) |> dev
    # Xv = Xv ./ sum(Xv, dims=1)
    a = abs.(F.U' * Xv)
    # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
    # batchV[:,1] =  reshape(c,:);
    batchV[:,1] =  reshape(a,:);
    
    
    for i in 2:t_samp
        Xh = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * Xv .+ J.b)))) |> dev 
        # Xh = Xh ./ sum(Xh, dims=1)
        a = abs.(F.V' * Xh)
        # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
        # batchH[:,i] =  reshape(c,:)
        batchH[:,i] =  reshape(a,:)
        
        Xv = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * Xh .+ J.a)))) |> dev 
        # Xv = Xv ./ sum(Xv, dims=1)
        a = abs.(F.U' * Xv)
        # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
        # batchV[:,i] = reshape(c,:);
        batchV[:,i] = reshape(a,:);
    end
    return batchV, batchH[:,2:end]
end

function amplitudes2(J, hparams; num=1, β=1.0, t_samp=40, ϵ=1e-5, dev=gpu )
    num=1
    Heff = H_effective(J,hparams; dev)
    # F = LinearAlgebra.svd(J.w, full=true)
    F = LinearAlgebra.svd(Heff, full=true)
    batchV = zeros(hparams.nv, t_samp)
    batchH = zeros(hparams.nh, t_samp)

    xh = sign.(rand(hparams.nh, num) .< 0.5) |> dev
    # xh = xh ./ sum(xh, dims=1)
    Xv = CuArray{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * xh .+ J.a)))) |> dev
    # Xv = Xv ./ sum(Xv, dims=1)
    a = abs.(F.U' * Xv)
    batchV[:,1] =  reshape(a,:);
    
    
    for i in 2:t_samp
        Xh = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * Xv .+ J.b)))) |> dev 
        # Xh = Xh ./ sum(Xh, dims=1)
        a = abs.(F.V' * Xh)
        # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
        # batchH[:,i] =  reshape(c,:)
        batchH[:,i] =  reshape(a,:)
        
        Xv = Array{Float32}(sign.(rand(hparams.nv, num) |> dev .< σ.(β .* (J.w * Xh .+ J.a)))) |> dev 
        # Xv = Xv ./ sum(Xv, dims=1)
        a = abs.(F.U' * Xv)
        # c = mean(a ./ (sum(a, dims=1) .+ dev(ϵ)), dims=2)
        # batchV[:,i] = reshape(c,:);
        batchV[:,i] = reshape(a,:);
    end
    return batchV, batchH[:,2:end]
end

function correlation(rbm, J, hparams; t_therm=10000, t_corr=10000, β=1)
    corr = []
    xh = sign.(rand(hparams.nh) .< 0.5) |> dev
    # xh = xh ./ sum(xh, dims=1)
    rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * xh .+ J.a)))) |> dev
    # rbm.v = rbm.v ./ sum(rbm.v, dims=1)

    for i in 1:t_therm
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
        # rbm.v = rbm.v ./ sum(rbm.v, dims=1)
    end

    rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
    # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
    m = σ.(β .* (J.w * rbm.h .+ J.a))
    rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< m)) |> dev
    # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
    
    Δv0 = m .- mean(m, dims=1)
    Δv = Δv0
    C0 = (Δv0' * Δv0)/hparams.nv
    Ct = (Δv0' * Δv)/hparams.nv
    append!(corr, Ct/C0)
    for i in 1:t_corr
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
        m = σ.(β .* (J.w * rbm.h .+ J.a))
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< m)) |> dev
        # rbm.v = rbm.v ./ sum(rbm.v, dims=1)
        Δv = m .- mean(m, dims=1)
        Ct = (Δv0' * Δv)/hparams.nv
        
        append!(corr, Ct/C0)
    end
    corr
end

function correlation2(rbm, J, hparams; t_therm=10000, t_corr=10000, β=1)
    corr = []
    xh = sign.(rand(hparams.nh) .< 0.5) |> dev
    # xh = xh ./ sum(xh, dims=1)
    rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * xh .+ J.a)))) |> dev
    # rbm.v = rbm.v ./ sum(rbm.v, dims=1)

    for i in 1:t_therm
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev
        # rbm.v = rbm.v ./ sum(rbm.v, dims=1)
    end

    rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev
    # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
    rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev 
    # rbm.v = rbm.v ./ sum(rbm.v, dims=1)
    m = rbm.v
    
    Δv0 = m .- mean(m, dims=1)
    Δv = Δv0
    C0 = (Δv0' * Δv0)/hparams.nv
    Ct = (Δv0' * Δv)/hparams.nv
    append!(corr, Ct/C0)
    for i in 1:t_corr
        rbm.h = Array{Float32}(sign.(rand(hparams.nh) |> dev .< σ.(β .* (J.w' * rbm.v .+ J.b)))) |> dev 
        # rbm.h = rbm.h ./ sum(rbm.h, dims=1)
        rbm.v = Array{Float32}(sign.(rand(hparams.nv) |> dev .< σ.(β .* (J.w * rbm.h .+ J.a)))) |> dev  
        # rbm.v = rbm.v ./ sum(rbm.v, dims=1)
        m = rbm.v
        Δv = m .- mean(m, dims=1)
        Ct = (Δv0' * Δv)/hparams.nv
        
        append!(corr, Ct/C0)
    end
    corr
end

###############Gaussian Spin Landscape

f(x,y,i, a0, b0, λ) = - (a0[i]*x + b0[i]*y + λ[i]*x*y )

function plotLandscape(s, t, a0, b0, λ, i=1, gibbs=0)
    p = plot(-15:10, -10:13, (x,y)->f(x,y,i, a0, b0, λ), st=:contourf, c=cgrad(:matter, 105, rev=true, scale = :exp, categorical=false), 
    xlabel="x", ylabel="y", clabels=true)
    p = plot!(s[i,:], t[i,:], markersize=7, markershapes = :circle, lw=0, markerstrokewidth=0, c=:blue, label="$i - $gibbs")
    p
end

function plotLandscape(rbm, J, a0, b0, λ, F, i=1, gibbs=0; β=1, num=5, TS, dev, hparams)
    p = plot(-15:10, -10:13, (x,y)->f(x,y,i, a0, b0, λ), st=:contour, fill=true, c=cgrad(:curl, 25, rev=false, categorical=true), 
    xlabel="x", ylabel="y", clabels=true)
    s = cpu(F.U' * rbm.v)
    t = cpu(F.Vt * rbm.h);
    p = plot!(s[i,:], t[i,:], markersize=4, markershapes = :circle, lw=0, markerstrokewidth=0.2, c=:black, label="$i - $gibbs")
    for j in 0:9
        idx = findall(x->x == j, TS.y)[1:num]
        vSamp = TS.x[:,idx] |> dev
        hSamp = Array{Float32}(sign.(rand(hparams.nh, num) |> dev .< σ.(β .* (J.w' * vSamp .+ J.b)))) |> dev ;
        
        s = cpu(F.U' * vSamp)
        t = cpu(F.Vt * hSamp);
        p = plot!(s[i,:], t[i,:], markersize=4, markershapes = :circle, lw=0, markerstrokewidth=0.2, c=:auto, label="$j")
    end
    p = plot!(legend = :outertopleft)
    p
end

function plotLandscapes(rbm, J, lnum ; τ=0, kmin=1, kmax=4, TS, dev, hparams)
    F = LinearAlgebra.svd(J.w, full=true);
    a0 = transpose(F.U) * J.a |> cpu
    b0 = F.Vt * J.b  |> cpu ;
    λ = F.S |> cpu ;

    fig = []
    for i in kmin:kmax
        push!(fig, plotLandscape(rbm, J, a0, b0, λ, F, i, τ; TS, dev, hparams))
    end
    fig = plot(fig..., size=(Int(500*(kmax-kmin)/2),Int(350*(kmax-kmin)/2)))
    
    # s = cpu(F.U' * rbm.v)
    # t = cpu(F.Vt * rbm.h);
    ###########################
    # if Gidx == 0
    #     fig = plotLandscape(s, t, a0, b0, λ, Gidx, τ)
    # else
    #     fig = []
    #     for k in 1:kmax
    #         push!(fig, plotLandscape(s, t, a0, b0, λ, k, τ))
    #     end
    #     fig = plot(fig..., size=(Int(350*kmax/2),Int(350*kmax/2)))
    # end
    ############################
    return fig
end

function saddlePointEnergy(J, hparams; dev)
    F = LinearAlgebra.svd(J.w, full=false);
    a0 = transpose(F.U) * J.a |> cpu
    b0 = F.Vt * J.b  |> cpu ;
    λ = F.S |> cpu ;
    sp_energy = sort(a0 .* b0 ./ (λ .+ 1.e-10))
    
    return sum(sp_energy)
end
    




