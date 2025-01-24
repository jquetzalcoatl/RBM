using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures
using CUDA
CUDA.device_reset!()
CUDA.device!(1)

include("../utils/init.jl")
# include("../scripts/exact_partition.jl")
include("../scripts/PhaseAnalysis.jl")
include("../scripts/langevin_doubleWell.jl")

PATH = "/home/javier/Projects/RBM/NewResults/"

modelName = config.model_analysis["files"][1]
modelName = modelName = "CD-500-T1-BW-replica1"
rbm, J, m, hparams, opt = loadModel(modelName, dev);
dict = loadDict(modelName)


function get_landscape_params(modelName::String, hparams::HyperParams, dict::Dict, l::Int=100)
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    nv=hparams.nv #28*28
    nh=hparams.nh #500

    s = size(readdir("$(dict["bdir"])/models/$(modelName)/J"),1)
    @info "$s models found"
    
    a0s = Array(zeros(nv,l))
    b0s = Array(zeros(nh,l))
    λs = Array(zeros(nh,l))
    μ_u = Array(zeros(nh,l))
    μ_w = Array(zeros(nh,l))
    # x_sdlpnt = Array(zeros(nh,l))
    # y_sdlpnt = Array(zeros(nh,l))

    x_s = Dict()
    y_s = Dict()
    u_s = Dict()
    w_s = Dict()
    R = Dict()
    θ = Dict()

    local u_gibbs, w_gibbs
        
    for i in 1:l
        idx = Int(s/l*i)
        @info i, idx
        J = load("$(dict["bdir"])/models/$(modelName)/J/J_$(idx).jld", "J")
        J.w = dev(J.w)
        J.b = dev(J.b)
        J.a = dev(J.a)
        F = LinearAlgebra.svd(J.w, full=true);

        μ_y = Array(F.Vt * dev(ones(hparams.nh)) * 0.5)
        μ_x = Array(F.U' * dev(ones(hparams.nv)) * 0.5)

        xsp = -Array(F.Vt * J.b) ./ Array(F.S)
        ysp = -Array(F.U' * J.a)[1:hparams.nh] ./ Array(F.S);
        # x_sdlpnt[:,i] = xsp
        # y_sdlpnt[:,i] = ysp

        for num_label in 0:9
            v = gpu(x_i[:,y_i .== num_label])
            h = sign.(rand(hparams.nh, size(v,2)) |> dev .< σ.(J.w' * v .+ J.b))

            x = cpu(F.U' * v)
            y = cpu(F.Vt * h);
            
            if i != 1
                R[string(num_label)] = cat(R[string(num_label)], .√ ((x[1:hparams.nh,:] .- xsp) .^2 .+ (y .- ysp) .^2), dims=3)
                θ[string(num_label)] = cat(θ[string(num_label)], atan.((y .- ysp) , (x[1:hparams.nh,:] .- xsp)), dims=3)
                x_s[string(num_label)] = cat(x_s[string(num_label)], x, dims=3)
                y_s[string(num_label)] = cat(y_s[string(num_label)], y, dims=3);
                u_s[string(num_label)] = cat(u_s[string(num_label)] , 1/sqrt(2) .* (x[1:hparams.nh] .+ y .- xsp .- ysp), dims=3)
                w_s[string(num_label)] = cat(w_s[string(num_label)] , 1/sqrt(2) .* (-x[1:hparams.nh] .+ y .+ xsp .- ysp), dims=3)
            else
                R[string(num_label)] = .√ ((x[1:hparams.nh,:] .- xsp) .^2 .+ (y .- ysp) .^2)
                θ[string(num_label)] = atan.((y .- ysp) , (x[1:hparams.nh,:] .- xsp))
                x_s[string(num_label)] = x
                y_s[string(num_label)] = y;
                u_s[string(num_label)] = 1/sqrt(2) .* (x[1:hparams.nh] .+ y .- xsp .- ysp)
                w_s[string(num_label)] = 1/sqrt(2) .* (-x[1:hparams.nh] .+ y .+ xsp .- ysp)
            end
        end

        λs[:,i] = Array(F.S)
        a0s[:,i] = Array(F.U' * J.a)
        b0s[:,i] = Array(F.Vt * J.b)
        μ_u[:,i] = 1/sqrt(2) .* (μ_x[1:hparams.nh] .+ μ_y .- xsp .- ysp)
        μ_w[:,i] = 1/sqrt(2) .* (-μ_x[1:hparams.nh] .+ μ_y .+ xsp .- ysp)

        v,h = gibbs_sample(J, hparams, 1000,500)
        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);
        if i!= 1
            u_gibbs = cat(u_gibbs , 1/sqrt(2) .* (x[1:hparams.nh] .+ y .- xsp .- ysp), dims=3)
            w_gibbs = cat(w_gibbs , 1/sqrt(2) .* (-x[1:hparams.nh] .+ y .+ xsp .- ysp), dims=3)
        else
            u_gibbs = 1/sqrt(2) .* (x[1:hparams.nh] .+ y .- xsp .- ysp)
            w_gibbs = 1/sqrt(2) .* (-x[1:hparams.nh] .+ y .+ xsp .- ysp)
        end
        
    end
    return λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs #, x_sdlpnt, y_sdlpnt
end

function get_landscape_params(hparams::HyperParams, l::Int=100)
    rbm, J, m, hparams, rbmZ = initModel(nv=hparams.nv, nh=hparams.nh, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    nv=hparams.nv #28*28
    nh=hparams.nh #500

    s = l
    @info "$s random models"
    
    a0s = Array(zeros(nv,l))
    b0s = Array(zeros(nh,l))
    λs = Array(zeros(nh,l))
    μ_u = Array(zeros(nh,l))
    μ_w = Array(zeros(nh,l))

    x_s = Dict()
    y_s = Dict()
    u_s = Dict()
    w_s = Dict()
    R = Dict()
    θ = Dict()

    local u_gibbs, w_gibbs
        
    for i in 1:l
        idx = Int(s/l*i)
        @info i, idx
        J.w = randn(size(J.w))
        J.a = randn(size(J.a))
        J.b = randn(size(J.b))
        J.w = dev(J.w)
        J.b = dev(J.b)
        J.a = dev(J.a)
        F = LinearAlgebra.svd(J.w, full=true);

        μ_y = Array(F.Vt * dev(ones(hparams.nh)) * 0.5)
        μ_x = Array(F.U' * dev(ones(hparams.nv)) * 0.5)

        xsp = -Array(F.Vt * J.b) ./ Array(F.S)
        ysp = -Array(F.U' * J.a)[1:hparams.nh] ./ Array(F.S);

        for num_label in 0:9
            v = gpu(x_i[:,y_i .== num_label])
            h = sign.(rand(hparams.nh, size(v,2)) |> dev .< σ.(J.w' * v .+ J.b))

            x = cpu(F.U' * v)
            y = cpu(F.Vt * h);
            
            if i != 1
                R[string(num_label)] = cat(R[string(num_label)], .√ ((x[1:hparams.nh,:] .- xsp) .^2 .+ (y .- ysp) .^2), dims=3)
                θ[string(num_label)] = cat(θ[string(num_label)], atan.((y .- ysp) , (x[1:hparams.nh,:] .- xsp)), dims=3)
                x_s[string(num_label)] = cat(x_s[string(num_label)], x, dims=3)
                y_s[string(num_label)] = cat(y_s[string(num_label)], y, dims=3);
                u_s[string(num_label)] = cat(u_s[string(num_label)] , 1/sqrt(2) .* (x[1:hparams.nh] .+ y .- xsp .- ysp), dims=3)
                w_s[string(num_label)] = cat(w_s[string(num_label)] , 1/sqrt(2) .* (-x[1:hparams.nh] .+ y .+ xsp .- ysp), dims=3)
            else
                R[string(num_label)] = .√ ((x[1:hparams.nh,:] .- xsp) .^2 .+ (y .- ysp) .^2)
                θ[string(num_label)] = atan.((y .- ysp) , (x[1:hparams.nh,:] .- xsp))
                x_s[string(num_label)] = x
                y_s[string(num_label)] = y;
                u_s[string(num_label)] = 1/sqrt(2) .* (x[1:hparams.nh] .+ y .- xsp .- ysp)
                w_s[string(num_label)] = 1/sqrt(2) .* (-x[1:hparams.nh] .+ y .+ xsp .- ysp)
            end
        end

        λs[:,i] = Array(F.S)
        a0s[:,i] = Array(F.U' * J.a)
        b0s[:,i] = Array(F.Vt * J.b)
        μ_u[:,i] = 1/sqrt(2) .* (μ_x[1:hparams.nh] .+ μ_y .- xsp .- ysp)
        μ_w[:,i] = 1/sqrt(2) .* (-μ_x[1:hparams.nh] .+ μ_y .+ xsp .- ysp)

        v,h = gibbs_sample(J, hparams, 1000,500)
        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);
        if i!= 1
            u_gibbs = cat(u_gibbs , 1/sqrt(2) .* (x[1:hparams.nh] .+ y .- xsp .- ysp), dims=3)
            w_gibbs = cat(w_gibbs , 1/sqrt(2) .* (-x[1:hparams.nh] .+ y .+ xsp .- ysp), dims=3)
        else
            u_gibbs = 1/sqrt(2) .* (x[1:hparams.nh] .+ y .- xsp .- ysp)
            w_gibbs = 1/sqrt(2) .* (-x[1:hparams.nh] .+ y .+ xsp .- ysp)
        end
        
    end
    return λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs
end

λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs = get_landscape_params(modelName, hparams, dict, 100)
λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs = get_landscape_params(hparams, 100)

f(x,y,i, a0, b0, λ) = - (a0[i]*x + b0[i]*y + λ[i]*x*y )
modelName
enLandIdx = 2
ep = 1
p = plot(-15:10, -10:13, (x,y)->f(x,y,enLandIdx, a0s[:,ep], b0s[:,ep], λs[:,ep]), st=:contourf, c=cgrad(:matter, 105, rev=true, scale = :exp, categorical=false), 
    xlabel="x", ylabel="y", clabels=true)
for num_label in string.(collect(0:9))
    plot!(x_s[num_label][enLandIdx,:,ep], y_s[num_label][enLandIdx,:,ep], st=:scatter, markerstrokewidth=0.1)
end
plot!([-b0s[enLandIdx,ep]/λs[enLandIdx,ep]],[-a0s[enLandIdx,ep]/λs[enLandIdx,ep]], ms=15, st=:scatter, c=:black, legend=:none, markershape=:x)

ep=100
plot()
for i in 1:8
    plot!(Θ["1"][i,:,ep], R["1"][i,:,ep], lw=0.01, opacity=0.2, proj = :polar, m = 4, markerstrokewidth=0.1, title="1", label="EL $i")
end
plot!()

plot()
for num_label in string.(collect(0:9))
    plot!(u_s[num_label][enLandIdx,:,ep], w_s[num_label][enLandIdx,:,ep], st=:scatter, markerstrokewidth=0.1)
end
plot!(u_gibbs[enLandIdx,:,ep], w_gibbs[enLandIdx,:,ep], st=:scatter, markerstrokewidth=0.1, markersize=6)
plot!()

function genAnimation(x_s, y_s, R, Θ, s, modelname)
    for num_label in string.(collect(0:9))
        anim = @animate for ep ∈ 1:size(R[num_label],3)
            plot()
            for i in 1:15
                plot!(Θ[num_label][i,:,ep], R[num_label][i,:,ep], proj = :polar, m = 4, markerstrokewidth=0.1, 
                    title="Label:$(num_label) \n Epoch: $(ep*s/size(R[num_label],3))", label="EL $i")
            end
            plot!(size=(550,550))
        end
        gif(anim, "$(PATH)/Figs/$(modelname)/anim_polar_$(num_label).gif", fps = 1)
    end
    
    for enLand in 1:10
        anim = @animate for ep ∈ 1:size(x_s[num_label],3)
            p = plot(-15:10, -10:13, (x,y)->f(x,y,enLandIdx, a0s[:,ep], b0s[:,ep], λs[:,ep]), 
                st=:contourf, c=cgrad(:matter, 105, rev=true, scale = :exp, categorical=false), xlabel="x", ylabel="y", clabels=true)
            for num_label in string.(collect(0:9))
                plot!(x_s[num_label][enLandIdx,:,ep], y_s[num_label][enLandIdx,:,ep], st=:scatter, markerstrokewidth=0.1)
            end
            plot!([-b0s[enLandIdx,ep]/λs[enLandIdx,ep]],[-a0s[enLandIdx,ep]/λs[enLandIdx,ep]], ms=15, st=:scatter, c=:black, legend=:none, markershape=:x)
            plot!(size=(550,550))
        end
        gif(anim, "$(PATH)/Figs/$(modelname)/anim_LS_$(num_label).gif", fps = 1)
    end
end

λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w

f_uw(u,w, a0, b0, λ) = a0[1:hparams.nh,:] .* b0 ./ λ .- λ .^2 .* (u .^ 2 .- w .^ 2) ./ 2

plot(μ_u', label=false, marker=:circle, markerstrokewidth=0.0, lw=2)
plot(reshape(mean(u_s["1"],dims=2),hparams.nh,:)', label=false, marker=:circle, markerstrokewidth=0.0, lw=2)
μ_u[idx,:]

begin
    idx=2
    label="0"
    p1 = hline([0], ls=:dash, color=:black, label="Saddle point", 
        ylabel="Position u", title="$(idx)")
    p1 = plot!(μ_u[idx,:], label="μ", marker=:circle, markerstrokewidth=0.0, markersize=8, lw=2)
    p1 = plot!(reshape(mean(u_s[label],dims=2),hparams.nh,:)[idx,:], label="Test data", marker=:circle, 
        markerstrokewidth=0.0, markersize=5, lw=2, ribbon=reshape(std(u_s[label],dims=2),hparams.nh,:)[idx,:])
    p1 = plot!(reshape(mean(u_gibbs,dims=2),hparams.nh,:)[idx,:], label="Gibbs data", marker=:circle, 
        markerstrokewidth=0.0, markersize=2, lw=2, ribbon=reshape(std(u_gibbs,dims=2),hparams.nh,:)[idx,:], frame=:box)

    p2 = hline([0], ls=:dash, color=:black, label="Saddle point",
        ylabel="Position w",)
    p2 = plot!(μ_w[idx,:], label="μ", marker=:circle, markerstrokewidth=0.0, markersize=8, lw=2)
    p2 = plot!(reshape(mean(w_s[label],dims=2),hparams.nh,:)[idx,:], label="Testing data", marker=:circle, 
        markerstrokewidth=0.0, markersize=5, lw=2, ribbon=reshape(std(w_s[label],dims=2),hparams.nh,:)[idx,:])
    p2 = plot!(reshape(mean(w_gibbs,dims=2),hparams.nh,:)[idx,:], label="Gibbs sampled data", marker=:circle, 
        markerstrokewidth=0.0, markersize=2, lw=2, ribbon=reshape(std(w_gibbs,dims=2),hparams.nh,:)[idx,:], frame=:box)

    # p3 = plot(f_uw(zeros(size(μ_u)),zeros(size(μ_w)), a0s, b0s, λs)[idx,:], 
        # ls=:dash, color=:black, label="Saddle point", ylabel="Position w", xlabel="Epoch")
    # p3 = plot!(f_uw(μ_u,μ_w, a0s, b0s, λs)[idx,:], label="μ", marker=:circle, markerstrokewidth=0.0, lw=2)
    # p3 = plot!(f_uw(reshape(mean(u_s[label],dims=2),hparams.nh,:),
        # reshape(mean(w_s["1"],dims=2),hparams.nh,:), a0s, b0s, λs)[idx,:],
        # label="Testing data", marker=:circle, 
        # markerstrokewidth=0.0, lw=2)
    # p3 = plot!(f_uw(reshape(mean(u_gibbs,dims=2),hparams.nh,:),
        # reshape(mean(w_gibbs,dims=2),hparams.nh,:), a0s, b0s, λs)[idx,:],
        # label="Gibbs sampled data", marker=:circle, 
        # markerstrokewidth=0.0, lw=2, frame=:box)
    p3 = plot(λs[idx,:], ls=:dash, color=:black, label="Sing value", ylabel="", xlabel="Epoch" )

    p = plot(p1,p2,p3, layout=(3,1), size=(600,900), left_margin=5mm)
    # savefig(p, PATH * "Symmetry_$(modelName)_$(idx).png")
    p
end


μ_u[idx,end]
λs[idx,end]
reshape(mean(u_s[label],dims=2),hparams.nh,:)[idx,end]

mean(u_s[label][1,:,end]  )^2
mean(u_s[label][1,:,end] .^ 2 )

using Roots

Λ = λs[1,end]
μ = 1.0
k = 0.5*hparams.nh
α = 0.001*hparams.nh

μ_u[:,end]
k=λs[1,end]^2 * mean(w_gibbs,dims=2)[1,1,end]/(μ_w[1,end] - mean(w_gibbs,dims=2)[1,1,end])

    
# plot(res, label=false)


#############
rbm, J, m, hparams, rbmZ = initModel(nv=hparams.nv, nh=hparams.nh, batch_size=500, lr=1.5, t=10, gpu_usage = false, optType="Adam")
Us = zeros(hparams.nv, hparams.nv, 100)
for i in 1:100
    J.w = randn(size(J.w))
    F = LinearAlgebra.svd(J.w, full=true);
    Us[:,:,i] = F.U
end

F = LinearAlgebra.svd(J.w, full=true);
-Array(F.Vt * J.b) ./ Array(F.S)

x_s["1"][1:hparams.nh,1,:] .- ones(hparams.nh)

plot(reshape(Us[2,:,:],:), st=:histogram, normalized=true)
plot!(-0.2:0.001:0.2, x-> 1/√(2π*0.035^2)*exp(-x^2/(2*0.035^2)), lw=2)

cov(u_s["1"][1,1:500,end], w_s["1"][1,1:500,end])
