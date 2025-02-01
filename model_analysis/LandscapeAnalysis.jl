using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures
using CUDA
CUDA.device_reset!()
CUDA.device!(4)

include("../utils/init.jl")
include("../scripts/PhaseAnalysis.jl")
include("../scripts/langevin_doubleWell.jl")

PATH = "/home/javier/Projects/RBM/NewResults/"

modelName = config.model_analysis["files"][1]
modelName = "CD-500-T1-BW-replica1"
modelName = "CD-FMNIST-500-T1000-BW-replica1-L"
modelName = "PCD-100-replica1"
modelName = "PCD-MNIST-500-lr-replica2"
rbm, J, m, hparams, opt = loadModel(modelName, dev, idx=100);
dict = loadDict(modelName)


function get_landscape_params(modelName::String, hparams::HyperParams, dict::Dict, l::Int=100)
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    nv=hparams.nv #28*28
    nh=hparams.nh #500
    L = min(hparams.nh, hparams.nv)

    s = size(readdir("$(dict["bdir"])/models/$(modelName)/J"),1)
    @info "$s models found"
    
    a0s = Array(zeros(nv,l+1))
    b0s = Array(zeros(nh,l+1))
    λs = Array(zeros(L,l+1))
    μ_u = Array(zeros(L,l+1))
    μ_w = Array(zeros(L,l+1))
    # x_sdlpnt = Array(zeros(nh,l))
    # y_sdlpnt = Array(zeros(nh,l))

    x_s = Dict()
    y_s = Dict()
    u_s = Dict()
    w_s = Dict()
    R = Dict()
    θ = Dict()

    local u_gibbs, w_gibbs, J
        
    for i in 0:l
        idx = Int(s/l*i)
        @info i, idx
        if i == 0
            rbm, J, m, hparams, rbmZ = initModel(nv=hparams.nv, nh=hparams.nh, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
        else
            J = load("$(dict["bdir"])/models/$(modelName)/J/J_$(idx).jld", "J")
        end
        J.w = dev(J.w)
        J.b = dev(J.b)
        J.a = dev(J.a)
        F = LinearAlgebra.svd(J.w, full=true);

        μ_y = Array(F.Vt * dev(ones(hparams.nh)) * 0.5)
        μ_x = Array(F.U' * dev(ones(hparams.nv)) * 0.5)
        

        xsp = -Array(F.Vt * J.b)[1:L] ./ Array(F.S)
        ysp = -Array(F.U' * J.a)[1:L] ./ Array(F.S);

        for num_label in 0:9
            v = gpu(x_i[:,y_i .== num_label])
            h = sign.(rand(hparams.nh, size(v,2)) |> dev .< σ.(J.w' * v .+ J.b))

            x = cpu(F.U' * v)
            y = cpu(F.Vt * h);
            
            if i != 0
                R[string(num_label)] = cat(R[string(num_label)], .√ ((x[1:L,:] .- xsp) .^2 .+ (y[1:L,:] .- ysp) .^2), dims=3)
                θ[string(num_label)] = cat(θ[string(num_label)], atan.((y[1:L,:] .- ysp) , (x[1:L,:] .- xsp)), dims=3)
                x_s[string(num_label)] = cat(x_s[string(num_label)], x, dims=3)
                y_s[string(num_label)] = cat(y_s[string(num_label)], y, dims=3);
                u_s[string(num_label)] = cat(u_s[string(num_label)] , 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp), dims=3)
                w_s[string(num_label)] = cat(w_s[string(num_label)] , 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp), dims=3)
            else
                R[string(num_label)] = .√ ((x[1:L,:] .- xsp) .^2 .+ (y[1:L,:] .- ysp) .^2)
                θ[string(num_label)] = atan.((y[1:L,:] .- ysp) , (x[1:L,:] .- xsp))
                x_s[string(num_label)] = x
                y_s[string(num_label)] = y;
                u_s[string(num_label)] = 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp)
                w_s[string(num_label)] = 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp)
            end
        end

        λs[:,i+1] = Array(F.S)
        a0s[:,i+1] = Array(F.U' * J.a)
        b0s[:,i+1] = Array(F.Vt * J.b)
        μ_u[:,i+1] = 1/sqrt(2) .* (μ_x[1:L] .+ μ_y[1:L] .- xsp .- ysp)
        μ_w[:,i+1] = 1/sqrt(2) .* (-μ_x[1:L] .+ μ_y[1:L] .+ xsp .- ysp)

        v,h = gibbs_sample(J, hparams, 1000,500)
        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);
        if i!= 0
            u_gibbs = cat(u_gibbs , 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp), dims=3)
            w_gibbs = cat(w_gibbs , 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp), dims=3)
        else
            u_gibbs = 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp)
            w_gibbs = 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp)
        end
        
    end
    return λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs #, x_sdlpnt, y_sdlpnt
end

function get_landscape_params(hparams::HyperParams, l::Int=100)
    rbm, J, m, hparams, rbmZ = initModel(nv=hparams.nv, nh=hparams.nh, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    nv=hparams.nv #28*28
    nh=hparams.nh #500
    L = min(hparams.nh, hparams.nv)

    s = l
    @info "$s random models"
    
    a0s = Array(zeros(nv,l))
    b0s = Array(zeros(nh,l))
    λs = Array(zeros(L,l))
    μ_u = Array(zeros(L,l))
    μ_w = Array(zeros(L,l))

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

        xsp = -Array(F.Vt * J.b)[1:L] ./ Array(F.S)
        ysp = -Array(F.U' * J.a)[1:L] ./ Array(F.S);

        for num_label in 0:9
            v = gpu(x_i[:,y_i .== num_label])
            h = sign.(rand(hparams.nh, size(v,2)) |> dev .< σ.(J.w' * v .+ J.b))

            x = cpu(F.U' * v)
            y = cpu(F.Vt * h);
            
            if i != 0
                R[string(num_label)] = cat(R[string(num_label)], .√ ((x[1:L,:] .- xsp) .^2 .+ (y[1:L,:] .- ysp) .^2), dims=3)
                θ[string(num_label)] = cat(θ[string(num_label)], atan.((y[1:L,:] .- ysp) , (x[1:L,:] .- xsp)), dims=3)
                x_s[string(num_label)] = cat(x_s[string(num_label)], x, dims=3)
                y_s[string(num_label)] = cat(y_s[string(num_label)], y, dims=3);
                u_s[string(num_label)] = cat(u_s[string(num_label)] , 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp), dims=3)
                w_s[string(num_label)] = cat(w_s[string(num_label)] , 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp), dims=3)
            else
                R[string(num_label)] = .√ ((x[1:L,:] .- xsp) .^2 .+ (y[1:L,:] .- ysp) .^2)
                θ[string(num_label)] = atan.((y[1:L,:] .- ysp) , (x[1:L,:] .- xsp))
                x_s[string(num_label)] = x
                y_s[string(num_label)] = y;
                u_s[string(num_label)] = 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp)
                w_s[string(num_label)] = 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp)
            end
        end

        λs[:,i+1] = Array(F.S)
        a0s[:,i+1] = Array(F.U' * J.a)
        b0s[:,i+1] = Array(F.Vt * J.b)
        μ_u[:,i+1] = 1/sqrt(2) .* (μ_x[1:L] .+ μ_y[1:L] .- xsp .- ysp)
        μ_w[:,i+1] = 1/sqrt(2) .* (-μ_x[1:L] .+ μ_y[1:L] .+ xsp .- ysp)

        v,h = gibbs_sample(J, hparams, 1000,500)
        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);
        if i!= 0
            u_gibbs = cat(u_gibbs , 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp), dims=3)
            w_gibbs = cat(w_gibbs , 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp), dims=3)
        else
            u_gibbs = 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp)
            w_gibbs = 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp)
        end
        
    end
    return λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs
end

λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs = get_landscape_params(modelName, hparams, dict, 100)
λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs = get_landscape_params(hparams, 100)

##################################################################################
##################################################################################
##################################################################################
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
##################################################################################
##################################################################################
##################################################################################


f_uw(u,w, a0, b0, λ) = a0[1:hparams.nh,:] .* b0 ./ λ .- λ .^2 .* (u .^ 2 .- w .^ 2) ./ 2

plot(μ_u', label=false, marker=:circle, markerstrokewidth=0.0, lw=2)
plot(reshape(mean(u_s["1"],dims=2),hparams.nh,:)', label=false, marker=:circle, markerstrokewidth=0.0, lw=2)
μ_u[idx,:]

begin
    idx=1
    label="0"
    L = min(hparams.nh, hparams.nv)
    p1 = hline([0], ls=:dash, color=:black, label="Saddle point", 
        ylabel="Position u", title="$(idx)")
    sgn = sign.(μ_w[idx,:])
    # sgn = ones(21)
    p1 = plot!(sgn .* μ_u[idx,:], label="μ", marker=:circle, markerstrokewidth=0.0, markersize=5, lw=2)
    p1 = plot!(sgn .* reshape(mean(u_s[label],dims=2),L,:)[idx,:], label="Test data", marker=:circle, 
        markerstrokewidth=0.0, markersize=5, lw=2, ribbon=reshape(std(u_s[label],dims=2),L,:)[idx,:])
    p1 = plot!(sgn .* reshape(mean(u_gibbs,dims=2),L,:)[idx,:], label="Gibbs data", marker=:circle, 
        markerstrokewidth=0.0, markersize=2, lw=2, ribbon=reshape(std(u_gibbs,dims=2),L,:)[idx,:], frame=:box, 
        legend = :outertopright) #, xlim=(30,40))
    #####
    # p1 = plot!([34],[(sgn .* reshape(mean(u_gibbs,dims=2),L,:)[idx,:])[34]], label="Gibbs data", marker=:square, 
    #     markerstrokewidth=0.0, markersize=5, lw=2, ribbon=reshape(std(u_gibbs,dims=2),L,:)[idx,:], frame=:box, 
    #     legend = :outertopright) #, xlim=(30,40))
    # p1 = plot!([32],[(sgn .* reshape(mean(u_gibbs,dims=2),L,:)[idx,:])[32]], label="Gibbs data", marker=:square, 
    #     markerstrokewidth=0.0, markersize=5, lw=2, ribbon=reshape(std(u_gibbs,dims=2),L,:)[idx,:], frame=:box, 
    #     legend = :outertopright) #, xlim=(30,40))
    # p1 = plot!([33],[(sgn .* reshape(mean(u_gibbs,dims=2),L,:)[idx,:])[33]], label="Gibbs data", marker=:square, 
    #     markerstrokewidth=0.0, markersize=5, lw=2, ribbon=reshape(std(u_gibbs,dims=2),L,:)[idx,:], frame=:box, 
    #     legend = :outertopright) #, xlim=(30,40))
    # p1 = plot!([35],[(sgn .* reshape(mean(u_gibbs,dims=2),L,:)[idx,:])[35]], label="Gibbs data", marker=:square, 
    #     markerstrokewidth=0.0, markersize=5, lw=2, ribbon=reshape(std(u_gibbs,dims=2),L,:)[idx,:], frame=:box, 
    #     legend = :outertopright) #, xlim=(30,40))

    p2 = hline([0], ls=:dash, color=:black, label="Saddle point",
        ylabel="Position w",)
    p2 = plot!(sgn .* μ_w[idx,:], label="μ", marker=:circle, markerstrokewidth=0.0, markersize=8, lw=2)
    p2 = plot!(sgn .* reshape(mean(w_s[label],dims=2),L,:)[idx,:], label="Test data", marker=:circle, 
        markerstrokewidth=0.0, markersize=5, lw=2, ribbon=reshape(std(w_s[label],dims=2),L,:)[idx,:])
    p2 = plot!(sgn .* reshape(mean(w_gibbs,dims=2),L,:)[idx,:], label="Gibbs sampled", marker=:circle, 
        markerstrokewidth=0.0, markersize=2, lw=2, ribbon=reshape(std(w_gibbs,dims=2),L,:)[idx,:], frame=:box,
        legend = :outertopright) #, xlim=(30,40))

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


sgn .* reshape(mean(u_s[label],dims=2),L,:)[idx,:]
sgn .* μ_u[idx,:]
(sgn .* reshape(mean(u_gibbs,dims=2),hparams.nh,:)[idx,:])[1:34]




170/5
35*5


##################
#########cholesky
J = load("$(dict["bdir"])/models/$(modelName)/J/J_1000.jld", "J")
J = Weights([getfield(J, field) |> dev for field in fieldnames(Weights)]...)

F = LinearAlgebra.svd(J.w, full=true);
delta = hparams.nh
v,h = gibbs_sample(J, hparams, 200,1000)

v,h = gibbs_sample(J, hparams, 2000,1)
v,h = gibbs_sample(v, J, hparams, size(v,2),1)

x = cpu(F.U' * v)
y = cpu(F.Vt * h);


z = vcat( hcat([y_s[lb][:,:,end] for lb in string.(collect(0:9))]...), hcat([x_s[lb][:,:,end] for lb in string.(collect(0:9))]...))
z = vcat( y,x)

cov1 = cov(z[1:2*hparams.nh,:]')
cov1 = (cov1 + cov1')/2
ChD_1 = cholesky(Hermitian(cov1));


# nsamples = 10000
nsamples = size(z,2)
z_samples = mean(z[1:2*hparams.nh,:], dims=2)[:] .+ ChD_1.L * randn(2*hparams.nh,nsamples);


# i=30
# plot(z[i,:], z[500+i,:], s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.1, frame=:box, label="MNIST data rot 2")
# plot!(z_samples[i,:], z_samples[500+i,:], s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.1, frame=:box, label="samples")

# plot(z[i,:], z_samples[i,:], s=:auto, markershapes = :circle, lw=0.0, markerstrokewidth=0.1, frame=:box, label="MNIST data rot 2")
# z
# z_samples[1,:] = z[1,:]

# for i in 1:100
#     z_samples[i,:] = z[i,:]
# end


figs = []
for i in 1:2
    if i == 1
        y_samples_unrot = z[1:delta,:]
        v_samples_fc = σ.(cpu(F.U) * vcat(cpu(Diagonal(F.S)), zeros(hparams.nv - hparams.nh, hparams.nh) ) * y_samples_unrot .+ cpu(J.a));
        v_samples_fc = Array{Float32}(sign.(rand(hparams.nv, size(v_samples_fc,2)) .< v_samples_fc));
    elseif i==2
        # nothing
        y_samples_unrot = z_samples[1:delta,:]
        v_samples_fc = σ.(cpu(F.U) * vcat(cpu(Diagonal(F.S)), zeros(hparams.nv - hparams.nh, hparams.nh) ) * y_samples_unrot .+ cpu(J.a));
        v_samples_fc = Array{Float32}(sign.(rand(hparams.nv, size(v_samples_fc,2)) .< v_samples_fc));
        v_samples_fc,h = gibbs_sample(v_samples_fc, J, hparams, size(v_samples_fc,2),1)
    end

    lnum=10
    mat = cat([cat([reshape(v_samples_fc[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
    mat_rot = reverse(transpose(mat), dims=1)
    push!(figs, heatmap(cpu(mat_rot)))
end
plot(figs..., size=(1200,900))

lnum=10
mat = cat([cat([reshape(v[:,i+j*lnum],28,28) for i in 1:lnum]..., dims=2) for j in 0:lnum-1]...,dims=1)
mat_rot = reverse(transpose(mat), dims=1)
heatmap(cpu(mat_rot))


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
