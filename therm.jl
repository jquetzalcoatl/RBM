using CUDA, Flux, JLD2
using Plots.PlotMeasures

include("utils/train.jl")

function loadLandscapes(PATH = "/home/javier/Projects/RBM/Results/",  modelname = "CD-500-T1000-5-BW-replica1-L"; l=30, nv=28*28, nh=500)
    s = size(readdir("$(PATH)/models/$(modelname)/J"),1)
    Us = Array(zeros(l))
    Σs = Array(zeros(l))
    Fs = Array(zeros(l))
    Zs = Array(zeros(l))
    
    UsRBM = Array(zeros(l))
    ΣsRBM = Array(zeros(l))
    FsRBM = Array(zeros(l))
    ZsRBM = Array(zeros(l))

    x_s = Dict()
    y_s = Dict()

    Δidx = s >= l ? Int(floor(s/l)) : 1
    for i in 1:min(l,s)
        idx = Δidx*i
        
        J = load("$(PATH)/models/$(modelname)/J/J_$(idx).jld", "J")
        J.w = gpu(J.w)
        J.b = gpu(J.b)
        J.a = gpu(J.a)
        F = LinearAlgebra.svd(J.w, full=true);

        v,h = data_val_samples(F)
        umean, wmean, σ_2u, σ_2w, a0, b0, λ = compute_stats(v, h, J)
        U, Σ, F, Z = compute_therm(umean, wmean, σ_2u, σ_2w, a0, b0, λ)

        Us[i] = U
        Σs[i] = Σ
        Fs[i] = F
        Zs[i] = Z
        
        v,h = gibbs_sampling(v,h,J)
        umean, wmean, σ_2u, σ_2w, a0, b0, λ = compute_stats(v, h, J)
        U, Σ, F, Z = compute_therm(umean, wmean, σ_2u, σ_2w, a0, b0, λ)
        
        UsRBM[i] = U
        ΣsRBM[i] = Σ
        FsRBM[i] = F
        ZsRBM[i] = Z
        
    end

    return Us, Σs, Fs, Zs, UsRBM, ΣsRBM, FsRBM, ZsRBM
end

function data_val_samples(F)
    # x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    
    # Compute avg in v-h space -> project to x-y and then to u-w
    λ = cpu(F.S);
    x_s = Dict()
    y_s = Dict()
    v_s = Dict()
    h_s = Dict()

    for num_label in 0:9
        v = gpu(x_i[:,y_i .== num_label])
        h = Array{Float32}(sign.(rand(500, size(v,2)) |> dev .< σ.(β .* (dev(J.w)' * v .+ dev(J.b))))) |> dev

        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);

        x_s[string(num_label)] = x
        y_s[string(num_label)] = y;

        v_s[string(num_label)] = cpu(v)
        h_s[string(num_label)] = cpu(h);
    end
    return hcat([v_s["$nl"] for nl in 0:9]...), hcat([h_s["$nl"] for nl in 0:9]...)
end


function compute_stats(v, h, J)
    J.w = cpu(J.w)
    J.b = cpu(J.b)
    J.a = cpu(J.a)
    v = cpu(v)
    h = cpu(h)
    F = LinearAlgebra.svd(J.w, full=true);
    λ = cpu(F.S)
    a0 = cpu(F.U' * J.a)[1:size(λ,1)]
    b0 = cpu(F.Vt * J.b);
    
    l = minimum([size(v,1), size(h,1)])
    v_mean = reshape(mean(v, dims=2),:)
    x_mean = (F.U' * v_mean)[1:l]

    h_mean = reshape(mean(h, dims=2),:)
    y_mean = (F.Vt * h_mean)

    x_σ = diag(F.U' * mean([v[:,i] .* v[:,i]' for i in 1:size(v,2)]) * F.U)[1:l]
    y_σ = diag(F.Vt * mean([h[:,i] .* h[:,i]' for i in 1:size(h,2)]) * F.V)
    xy = diag(F.Vt * mean([h[:,i] .* v[:,i]' for i in 1:size(h,2)]) * F.U)
    xsp = - cpu(F.Vt * J.b) ./ λ
    ysp = - cpu(F.U' * J.a)[1:l] ./ λ

    umean = 1/√2 .* (cpu(y_mean + x_mean) - ysp - xsp)
    u_σ =  1/2 .* (cpu(y_σ + x_σ) + (ysp+xsp) .^2 + 2 .* cpu(xy) - 2 .* cpu(x_mean + y_mean) .* (xsp + ysp) )
    σ_2u = u_σ .- umean .^ 2

    wmean = 1/√2 .* (cpu(y_mean - x_mean) - ysp + xsp)
    w_σ =  1/2 .* (cpu(y_σ + x_σ) + (ysp-xsp) .^2 - 2 .* cpu(xy) - 2 .* cpu(y_mean - x_mean ) .* (ysp - xsp) )
    σ_2w = w_σ .- wmean .^ 2 ;
    
    return umean, wmean, σ_2u, σ_2w, a0, b0, λ
end

function compute_therm(umean, wmean, σ_2u, σ_2w, a0, b0, λ)
    # U = sum(a0 .* b0 ./ λ .+ 1/2 .* (umean .^ 2 ./ (σ_2u .* (1 .+ λ .* σ_2u)) .+  wmean .^ 2 ./ (σ_2w .* (1 .- λ .* σ_2w)))) - size(λ,1)
    U = sum(a0 .* b0 ./ λ .- λ/2 .* (umean .^ 2 ./ (1 .+ λ .* σ_2u) .-  wmean .^ 2 ./ (1 .- λ .* σ_2w))) - size(λ,1)
    Σ = sum(log.( σ_2u .* σ_2w))/2 + size(λ,1)*(log(2π) - 1)
    F = U - Σ
    return U, Σ, F, exp(-F)
end

function gibbs_sampling(v,h,J)
    β=1
    dev = gpu
    nh = size(h,1)
    nv = size(v,1)
    num= minimum([size(v,2), 1000])
    v = gpu(v[:,num])
    h = gpu(h[:,num])
    J.w = gpu(J.w)
    J.b = gpu(J.b)
    J.a = gpu(J.a)
    
    for i in 1:5000
        h = Array{Float32}(sign.(rand(nh, num) |> dev .< σ.(β .* (J.w' * v .+ J.b)))) |> dev
        v = Array{Float32}(sign.(rand(nv, num) |> dev .< σ.(β .* (J.w * h .+ J.a)))) |> dev 
    end
    return cpu(v),cpu(h)
end

function saveModePlot(Us, Σs, Fs, Zs, UsRBM, ΣsRBM, FsRBM, ZsRBM, modelname)
    isdir("$(PATH)/Figs/$(modelname)") || mkpath("$(PATH)/Figs/$(modelname)")
    f = plot(Us, lw=2.5, label="Data")
    f = plot!(UsRBM, lw=2.5, label="RBM")
    f = plot!(size=(700,500), xlabel="Epochs (x10)", ylabel="Internal Energy", frame=:box, margin = 15mm)
    savefig(f, "$(PATH)/Figs/$(modelname)/int_energy_$(modelname).png")
    
    f = plot(Σs, lw=2.5, label="Data")
    f = plot!(ΣsRBM, lw=2.5, label="RBM")
    f = plot!(size=(700,500), xlabel="Epochs (x10)", ylabel="Entropy", frame=:box, margin = 15mm)
    savefig(f, "$(PATH)/Figs/$(modelname)/entropy_$(modelname).png")
    
    f = plot(Fs, lw=2.5, label="Data")
    f = plot!(FsRBM, lw=2.5, label="RBM")
    f = plot!(size=(700,500), xlabel="Epochs (x10)", ylabel="Free Energy", frame=:box, margin = 15mm)
    savefig(f, "$(PATH)/Figs/$(modelname)/free_energy_$(modelname).png")
    
    jldsave("$(PATH)/Figs/$(modelname)/thermoParams.jld", sdata=Σs, edata=Us, fdata=Fs, srbm=ΣsRBM, erbm=UsRBM, frbm=FsRBM)
    
end

PATH = "/home/javier/Projects/RBM/Results/"
l=100
nv=28*28
nh=500
dev = gpu
β = 1.0
modelName = "CD-500-T1000-5-BW-replica1-L"
rbm, J, m, hparams, opt = loadModel(modelName, gpu);
x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
for i in 1:5
    # modelname = "Rdm-500-T1-replica$(i)"
    # modelname = "Rdm-500-T1-BW-replica$(i)"
    # modelname = "CD-500-T1-replica$(i)"
    # modelname = "CD-500-T1-BW-replica$(i)"
    # modelname = "CD-500-T10-BW-replica$(i)"
    # modelname = "Rdm-500-T10-BW-replica$(i)"
    # modelname = "CD-500-T100-BW-replica$(i)"
    # modelname = "Rdm-500-T100-BW-replica$(i)"
    
    # modelname = "CD-500-T1000-5-BW-replica$(i)-L"
    modelname = "PCD-500-replica$(i)"
    Us, Σs, Fs, Zs, UsRBM, ΣsRBM, FsRBM, ZsRBM = loadLandscapes(PATH, modelname; l, nv, nh);
    
    saveModePlot(Us, Σs, Fs, Zs, UsRBM, ΣsRBM, FsRBM, ZsRBM, modelname)
end