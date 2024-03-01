using CUDA, Flux, JLD2
using Plots.PlotMeasures

include("utils/train.jl")

# function loadLandscapes(x_i, y_i, hparams, PATH = "/home/javier/Projects/RBM/Results/",  modelname = "CD-500-T1000-5-BW-replica1-L"; l=30, nv=28*28, nh=500)
#     s = size(readdir("$(PATH)/models/$(modelname)/J"),1)
#     Us = Array(zeros(l))
#     Σs = Array(zeros(l))
#     Fs = Array(zeros(l))
#     Zs = Array(zeros(l))
    
#     UsRBM = Array(zeros(l))
#     ΣsRBM = Array(zeros(l))
#     FsRBM = Array(zeros(l))
#     ZsRBM = Array(zeros(l))

#     x_s = Dict()
#     y_s = Dict()

#     Δidx = s >= l ? Int(floor(s/l)) : 1
#     for i in 1:min(l,s)
#         idx = Δidx*i
        
#         J = load("$(PATH)/models/$(modelname)/J/J_$(idx).jld", "J")
#         J.w = gpu(J.w)
#         J.b = gpu(J.b)
#         J.a = gpu(J.a)
#         F = LinearAlgebra.svd(J.w, full=true);

#         v,h = data_val_samples(F, J, x_i, y_i; hparams)
#         umean, wmean, σ_2u, σ_2w, a0, b0, λ = compute_stats(v, h, J)
#         U, Σ, F, Z = compute_therm(umean, wmean, σ_2u, σ_2w, a0, b0, λ)

#         Us[i] = U
#         Σs[i] = Σ
#         Fs[i] = F
#         Zs[i] = Z
        
#         v,h = gibbs_sampling(v,h,J)
#         umean, wmean, σ_2u, σ_2w, a0, b0, λ = compute_stats(v, h, J)
#         U, Σ, F, Z = compute_therm(umean, wmean, σ_2u, σ_2w, a0, b0, λ)
        
#         UsRBM[i] = U
#         ΣsRBM[i] = Σ
#         FsRBM[i] = F
#         ZsRBM[i] = Z
        
#     end

#     return Us, Σs, Fs, Zs, UsRBM, ΣsRBM, FsRBM, ZsRBM
# end

function loadLandscapes(x_i, y_i, hparams, PATH = "/home/javier/Projects/RBM/Results/",  modelname = "CD-500-T1000-5-BW-replica1-L"; l=30, nv=28*28, nh=500)
    s = size(readdir("$(PATH)/models/$(modelname)/J"),1)
    
    Fs = Array(zeros(l))
    
    FsRBM = Array(zeros(l))

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

        # v,h = data_val_samples(F, J, x_i, y_i; hparams)
        v,h, x_val,y_val = data_val_samples(F, J, x_i, y_i; hparams)
        z, λ, μ, ϵ, G, B, γ, C = exp_arg(x_val,y_val, F, J, hparams)
        logZ = eff_log_partition_function(λ, μ, ϵ, G, B, hparams)
        # umean, wmean, σ_2u, σ_2w, a0, b0, λ = compute_stats(v, h, J)
        # U, Σ, F, Z = compute_therm(umean, wmean, σ_2u, σ_2w, a0, b0, λ)

        Fs[i] = -logZ
        
        v,h = gibbs_sampling(v,h,J)
        z, λ, μ, ϵ, G, B, γ, C = exp_arg(x_val,y_val, F, J, hparams)
        logZ = eff_log_partition_function(λ, μ, ϵ, G, B, hparams)
        # umean, wmean, σ_2u, σ_2w, a0, b0, λ = compute_stats(v, h, J)
        # U, Σ, F, Z = compute_therm(umean, wmean, σ_2u, σ_2w, a0, b0, λ)
        
        FsRBM[i] = -logZ
        
    end

    return Fs, FsRBM
end

function data_val_samples(F, J, x_i, y_i; hparams)
    # x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    
    # Compute avg in v-h space -> project to x-y and then to u-w
    λ = cpu(F.S);
    x_s = Dict()
    y_s = Dict()
    v_s = Dict()
    h_s = Dict()

    for num_label in 0:9
        v = gpu(x_i[:,y_i .== num_label])
        h = Array{Float32}(sign.(rand(hparams.nh, size(v,2)) |> dev .< σ.(β .* (dev(J.w)' * v .+ dev(J.b))))) |> dev

        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);

        x_s[string(num_label)] = x
        y_s[string(num_label)] = y;

        v_s[string(num_label)] = cpu(v)
        h_s[string(num_label)] = cpu(h);
    end
    return hcat([v_s["$nl"] for nl in 0:9]...), hcat([h_s["$nl"] for nl in 0:9]...), hcat([x_s["$nl"] for nl in 0:9]...), hcat([y_s["$nl"] for nl in 0:9]...)
end

function data_val_samples(F; avg=false)
    # x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    
    # Compute avg in v-h space -> project to x-y and then to u-w
    λ = cpu(F.S);
    x_s = Dict()
    y_s = Dict()
    v_s = Dict()
    h_s = Dict()

    for num_label in 0:9
        v = gpu(x_i[:,y_i .== num_label])
        if avg
            h = σ.(β .* (dev(J.w)' * v .+ dev(J.b))) |> dev
        else
            h = Array{Float32}(sign.(rand(hparams.nh, size(v,2)) |> dev .< σ.(β .* (dev(J.w)' * v .+ dev(J.b))))) |> dev
        end

        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);

        x_s[string(num_label)] = x
        y_s[string(num_label)] = y;

        v_s[string(num_label)] = cpu(v)
        h_s[string(num_label)] = cpu(h);
    end
    return hcat([v_s["$nl"] for nl in 0:9]...), hcat([h_s["$nl"] for nl in 0:9]...), hcat([x_s["$nl"] for nl in 0:9]...), hcat([y_s["$nl"] for nl in 0:9]...)
end


function exp_arg(x_val,y_val, F, J, hparams)
    λ = cpu(F.S)
    
    nh = min(size(y_val,1), size(x_val,1))
    nv = max(size(y_val,1), size(x_val,1))

    z = size(y_val,1) <= size(x_val,1) ? cat(y_val, x_val, dims=1) : cat(x_val, y_val, dims=1)

    μ = reshape(mean(z, dims=2), :)
    C = cov(z[1:2*nh,:]')
    Cinv = inv(C)
    # @assert C * Cinv ≈ I

    mat_size = 2*nh
    G = zeros(mat_size,mat_size)
    for j in mat_size-nh+1:mat_size
        i = j-nh
        G[i,j] = ( i <= nh ? λ[i] : 0 ) # 1.0
    end
    G = G + transpose(G)
    G = (Cinv+Cinv')/2 + G;
    # G = Cinv + G;
    
    a0 = F.U' * J.a
    b0 = F.Vt * J.b
    c = size(b0,1) <= size(a0,1) ? cpu(cat(b0,a0, dims=1)) : cpu(cat(a0,b0, dims=1))

    ϵ2 = zeros(mat_size)
    ϵ2[1:nh] = [μ[nh + i]*λ[i] for i in 1:nh ]
    ϵ2[end-nh+1:end] = [μ[i]*λ[i] for i in 1:nh ]

    if nv > nh
        ϵ = c + cat(ϵ2, zeros(nv - nh), dims=1);
        B = inv(cov(z[2*nh+1:end,:]'));
    else
        ϵ = c + ϵ2;
        B = 0;
    end
    return z, λ, μ, ϵ, G, B, c, C
end

function eff_log_partition_function(λ, μ, ϵ, G, B, hparams)
    nh = Int(size(G,1)/2)
    nv = typeof(B) != Int64 ? size(B,1) + nh : nh
    if typeof(B) != Int64
        return 0.5 * (ϵ[1:2*nh]' * inv(G) * ϵ[1:2*nh] + ϵ[2*nh+1:end]' * inv(B) * ϵ[2*nh+1:end]) + μ' * ϵ + nv/2 * log(2π) - sum([μ[i] * μ[i+ nh] * λ[i] for i in 1:nh]) - 0.5 * (sum(log.(eigen(G).values[eigen(G).values .> 0])) + sum(log.(eigen(B).values)))
    else
        return 0.5 * (ϵ[1:2*nh]' * inv(G) * ϵ[1:2*nh]) + μ' * ϵ + nv/2 * log(2π) - sum([μ[i] * μ[i+ nh] * λ[i] for i in 1:nh]) - 0.5 * (sum(log.(eigen(G).values[eigen(G).values .> 0])) )
    end
end


# function compute_stats(v, h, J)
#     J.w = cpu(J.w)
#     J.b = cpu(J.b)
#     J.a = cpu(J.a)
#     v = cpu(v)
#     h = cpu(h)
#     F = LinearAlgebra.svd(J.w, full=true);
#     λ = cpu(F.S)
#     a0 = cpu(F.U' * J.a)[1:size(λ,1)]
#     b0 = cpu(F.Vt * J.b);
    
#     l = minimum([size(v,1), size(h,1)])
#     v_mean = reshape(mean(v, dims=2),:)
#     x_mean = (F.U' * v_mean)[1:l]

#     h_mean = reshape(mean(h, dims=2),:)
#     y_mean = (F.Vt * h_mean)

#     x_σ = diag(F.U' * mean([v[:,i] .* v[:,i]' for i in 1:size(v,2)]) * F.U)[1:l]
#     y_σ = diag(F.Vt * mean([h[:,i] .* h[:,i]' for i in 1:size(h,2)]) * F.V)
#     xy = diag(F.Vt * mean([h[:,i] .* v[:,i]' for i in 1:size(h,2)]) * F.U)
#     xsp = - cpu(F.Vt * J.b) ./ λ
#     ysp = - cpu(F.U' * J.a)[1:l] ./ λ

#     umean = 1/√2 .* (cpu(y_mean + x_mean) - ysp - xsp)
#     u_σ =  1/2 .* (cpu(y_σ + x_σ) + (ysp+xsp) .^2 + 2 .* cpu(xy) - 2 .* cpu(x_mean + y_mean) .* (xsp + ysp) )
#     σ_2u = u_σ .- umean .^ 2

#     wmean = 1/√2 .* (cpu(y_mean - x_mean) - ysp + xsp)
#     w_σ =  1/2 .* (cpu(y_σ + x_σ) + (ysp-xsp) .^2 - 2 .* cpu(xy) - 2 .* cpu(y_mean - x_mean ) .* (ysp - xsp) )
#     σ_2w = w_σ .- wmean .^ 2 ;
    
#     return umean, wmean, σ_2u, σ_2w, a0, b0, λ
# end

# function compute_therm(umean, wmean, σ_2u, σ_2w, a0, b0, λ; TOL=0.05)
#     u_softTerm = sign.(1 .- σ_2u .* λ) * TOL
#     w_softTerm = sign.(1 .- σ_2w .* λ) * TOL
#     # U = sum(a0 .* b0 ./ λ .+ 1/2 .* (umean .^ 2 ./ (σ_2u .* (1 .+ λ .* σ_2u)) .+  wmean .^ 2 ./ (σ_2w .* (1 .- λ .* σ_2w)))) - size(λ,1)
#     U = sum(a0 .* b0 ./ λ .- λ/2 .* (umean .^ 2 ./ (1 .+ λ .* σ_2u .+ u_softTerm) .-  wmean .^ 2 ./ (1 .- λ .* σ_2w .+ w_softTerm))) + size(λ,1)
#     Σ = sum(log.( σ_2u .* σ_2w))/2 + size(λ,1)*(log(2π) + 1)
#     F = U - Σ
#     return U, Σ, F, exp(-F)
# end

function gibbs_sampling(v,h,J; mcs=5000, dev0=cpu)
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
    
    for i in 1:mcs
        h = Array{Float32}(sign.(rand(nh, num) |> dev .< σ.(β .* (J.w' * v .+ J.b)))) |> dev
        v = Array{Float32}(sign.(rand(nv, num) |> dev .< σ.(β .* (J.w * h .+ J.a)))) |> dev 
    end
    return dev0(v),dev0(h)
end

function saveModePlot(Fs, FsRBM, modelname)
    isdir("$(PATH)/Figs/$(modelname)") || mkpath("$(PATH)/Figs/$(modelname)")
#     f = plot(Us, lw=2.5, label="Data")
#     f = plot!(UsRBM, lw=2.5, label="RBM")
#     f = plot!(size=(700,500), xlabel="Epochs (x10)", ylabel="Internal Energy", frame=:box, margin = 15mm)
#     savefig(f, "$(PATH)/Figs/$(modelname)/int_energy_$(modelname).png")
    
#     f = plot(Σs, lw=2.5, label="Data")
#     f = plot!(ΣsRBM, lw=2.5, label="RBM")
#     f = plot!(size=(700,500), xlabel="Epochs (x10)", ylabel="Entropy", frame=:box, margin = 15mm)
#     savefig(f, "$(PATH)/Figs/$(modelname)/entropy_$(modelname).png")
    
    f = plot(Fs, lw=2.5, label="Data")
    f = plot!(FsRBM, lw=2.5, label="RBM")
    f = plot!(size=(700,500), xlabel="Epochs (x10)", ylabel="Free Energy", frame=:box, margin = 15mm)
    savefig(f, "$(PATH)/Figs/$(modelname)/free_energy_$(modelname).png")
    
    # jldsave("$(PATH)/Figs/$(modelname)/thermoParams.jld", sdata=Σs, edata=Us, fdata=Fs, srbm=ΣsRBM, erbm=UsRBM, frbm=FsRBM)
    jldsave("$(PATH)/Figs/$(modelname)/thermoParams.jld", fdata=Fs, frbm=FsRBM)
    
end


if abspath(PROGRAM_FILE) == @__FILE__
    PATH = "/home/javier/Projects/RBM/Results/"
    l=100
    # nv=28*28
    # nh=500
    dev = gpu
    β = 1.0
    # modelName = "CD-500-T1000-5-BW-replica1-L"
    

    # for model in ["Rdm-500-T10-BW-replica", "Rdm-500-T100-BW-replica", "CD-500-T1-replica", "CD-500-T1-BW-replica", "CD-500-T10-BW-replica", "CD-500-T100-BW-replica", "CD-500-T1000-5-BW-replica-L", "PCD-500-replica", "PCD-100-replica"]
    # for model in ["CD-500-T1000-5-BW-replica-L", "PCD-500-replica", "PCD-100-replica"]
    for model in ["PCD-500-3000-replica", "PCD-500-1200-replica", "PCD-500-784-replica"]
        modelName = model * "1"
        rbm, J, m, hparams, opt = loadModel(modelName, gpu);
        x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
        for i in 1:5
            if model != "CD-500-T1000-5-BW-replica-L"
                modelname = model * "$(i)"
            else
                modelname = "CD-500-T1000-5-BW-replica$(i)-L"
            end
            @info modelname
            Fs, FsRBM = loadLandscapes(x_i, y_i, hparams, PATH, modelname; l, hparams.nv, hparams.nh);

            saveModePlot(Fs, FsRBM, modelname)
        end
    end  
end