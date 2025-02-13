using Random, Plots, Statistics, LinearAlgebra, Plots.PlotMeasures
using CUDA
CUDA.device_reset!()
CUDA.device!(0)

include("../utils/init.jl")
include("../scripts/PhaseAnalysis.jl")
include("../scripts/langevin_doubleWell.jl")

PATH = "/home/javier/Projects/RBM/NewResults/"

modelName = config.model_analysis["files"][6]
modelName = "CD-500-T1-BW-replica1"
modelName = "CD-FMNIST-500-T1000-BW-replica1-L"
modelName = "PCD-100-replica1"
modelName = "PCD-MNIST-500-lr-replica2"
rbm, J, m, hparams, opt = loadModel(modelName, dev, idx=100);
dict = loadDict(modelName)


function get_landscape_params(modelName::String, hparams::HyperParams, dict::Dict, l::Int=100, num_chains::Int=1000; seq::Bool=false)
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    nv=hparams.nv #28*28
    nh=hparams.nh #500
    L = min(hparams.nh, hparams.nv)

    s = size(readdir("$(dict["bdir"])/models/$(modelName)/J"),1)
    @info "$s models found"
    
    a0s = Array{Float64}(undef, nv,l+1) #Array(zeros(nv,l+1))
    b0s = Array{Float64}(undef, nh,l+1) #Array(zeros(nh,l+1))
    λs = Array{Float64}(undef, L,l+1) #Array(zeros(L,l+1))
    μ_u = Array{Float64}(undef, L,l+1) #Array(zeros(L,l+1))
    μ_w = Array{Float64}(undef, L,l+1) #Array(zeros(L,l+1))

    x_s = Dict()
    y_s = Dict()
    u_s = Dict()
    w_s = Dict()
    R = Dict()
    θ = Dict()
    for i in 0:9
        num_label = size(findall(x->x==i,y_i),1)
        x_s[string(i)] = Array{Float64}(undef, hparams.nv,num_label,l+1) # zeros(hparams.nv,num_label,l+1 )
        y_s[string(i)] = Array{Float64}(undef, hparams.nh,num_label,l+1) #zeros(hparams.nh,num_label,l+1 )
        u_s[string(i)] = Array{Float64}(undef, L,num_label,l+1) #zeros(L,num_label,l+1 )
        w_s[string(i)] = Array{Float64}(undef, L,num_label,l+1) #zeros(L,num_label,l+1 )
        R[string(i)] = Array{Float64}(undef, L,num_label,l+1) #zeros(L,num_label,l+1 )
        θ[string(i)] = Array{Float64}(undef,L,num_label,l+1) #zeros(L,num_label,l+1 )
    end

    local J
    u_gibbs = Array{Float64}(undef, L,num_chains,l+1 ) #zeros(L,num_chains,l+1 )
    w_gibbs = Array{Float64}(undef, L,num_chains,l+1 ) #zeros(L,num_chains,l+1 )
        
    for i in 0:l
        idx = seq ? i : Int(s/l*i)
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

            R[string(num_label)][:,:,i+1] = .√ ((x[1:L,:] .- xsp) .^2 .+ (y[1:L,:] .- ysp) .^2)
            θ[string(num_label)][:,:,i+1] = atan.((y[1:L,:] .- ysp) , (x[1:L,:] .- xsp))
            x_s[string(num_label)][:,:,i+1] = x
            y_s[string(num_label)][:,:,i+1] = y;
            u_s[string(num_label)][:,:,i+1] = 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp)
            w_s[string(num_label)][:,:,i+1] = 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp)
        end

        λs[:,i+1] = Array(F.S)
        a0s[:,i+1] = Array(F.U' * J.a)
        b0s[:,i+1] = Array(F.Vt * J.b)
        μ_u[:,i+1] = 1/sqrt(2) .* (μ_x[1:L] .+ μ_y[1:L] .- xsp .- ysp)
        μ_w[:,i+1] = 1/sqrt(2) .* (-μ_x[1:L] .+ μ_y[1:L] .+ xsp .- ysp)

        v,h = gibbs_sample(J, hparams, num_chains,500)
        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);
        u_gibbs[:,:,i+1] = 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp)
        w_gibbs[:,:,i+1] = 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp)
    end
    return λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs
end

function get_landscape_params(hparams::HyperParams, l::Int=100, num_chains::Int=1000)
    rbm, J, m, hparams, rbmZ = initModel(nv=hparams.nv, nh=hparams.nh, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    nv=hparams.nv #28*28
    nh=hparams.nh #500
    L = min(hparams.nh, hparams.nv)

    s = l
    @info "$(s+1) random models"
    
    a0s = Array(zeros(nv,l+1))
    b0s = Array(zeros(nh,l+1))
    λs = Array(zeros(L,l+1))
    μ_u = Array(zeros(L,l+1))
    μ_w = Array(zeros(L,l+1))

    x_s = Dict()
    y_s = Dict()
    u_s = Dict()
    w_s = Dict()
    R = Dict()
    θ = Dict()
    for i in 0:9
        num_label = size(findall(x->x==i,y_i),1)
        x_s[string(i)] = zeros(hparams.nv,num_label,l+1 )
        y_s[string(i)] = zeros(hparams.nh,num_label,l+1 )
        u_s[string(i)] = zeros(L,num_label,l+1 )
        w_s[string(i)] = zeros(L,num_label,l+1 )
        R[string(i)] = zeros(L,num_label,l+1 )
        θ[string(i)] = zeros(L,num_label,l+1 )
    end

    u_gibbs = zeros(L,num_chains,l+1 )
    w_gibbs = zeros(L,num_chains,l+1 )
        
    for i in 0:l
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
            
            R[string(num_label)][:,:,i+1] = .√ ((x[1:L,:] .- xsp) .^2 .+ (y[1:L,:] .- ysp) .^2)
            θ[string(num_label)][:,:,i+1] = atan.((y[1:L,:] .- ysp) , (x[1:L,:] .- xsp))
            x_s[string(num_label)][:,:,i+1] = x
            y_s[string(num_label)][:,:,i+1] = y;
            u_s[string(num_label)][:,:,i+1] = 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp)
            w_s[string(num_label)][:,:,i+1] = 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp)
        end

        λs[:,i+1] = Array(F.S)
        a0s[:,i+1] = Array(F.U' * J.a)
        b0s[:,i+1] = Array(F.Vt * J.b)
        μ_u[:,i+1] = 1/sqrt(2) .* (μ_x[1:L] .+ μ_y[1:L] .- xsp .- ysp)
        μ_w[:,i+1] = 1/sqrt(2) .* (-μ_x[1:L] .+ μ_y[1:L] .+ xsp .- ysp)

        v,h = gibbs_sample(J, hparams, num_chains,500)
        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);
        u_gibbs[:,:,i+1] = 1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp)
        w_gibbs[:,:,i+1] = 1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp)
    end
    return λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs
end

function getm_std(arr::Matrix{Float32})
    m = mean(arr,dims=2)
    sd = std(arr,dims=2)
    return cat(m,sd, dims=2)
end

function get_landscape_params_mstd(modelName::String, hparams::HyperParams, dict::Dict, l::Int=100, num_chains::Int=1000; seq::Bool=false)
    x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    nv=hparams.nv #28*28
    nh=hparams.nh #500
    L = min(hparams.nh, hparams.nv)

    s = size(readdir("$(dict["bdir"])/models/$(modelName)/J"),1)
    @info "$s models found"
    
    a0s = Array{Float64}(undef, nv,l+1) #Array(zeros(nv,l+1))
    b0s = Array{Float64}(undef, nh,l+1) #Array(zeros(nh,l+1))
    λs = Array{Float64}(undef, L,l+1) #Array(zeros(L,l+1))
    μ_u = Array{Float64}(undef, L,l+1) #Array(zeros(L,l+1))
    μ_w = Array{Float64}(undef, L,l+1) #Array(zeros(L,l+1))

    x_s = Dict()
    y_s = Dict()
    u_s = Dict()
    w_s = Dict()
    R = Dict()
    θ = Dict()
    for i in 0:9
        num_label = 2 # size(findall(x->x==i,y_i),1)
        x_s[string(i)] = Array{Float64}(undef, hparams.nv,num_label,l+1) # zeros(hparams.nv,num_label,l+1 )
        y_s[string(i)] = Array{Float64}(undef, hparams.nh,num_label,l+1) #zeros(hparams.nh,num_label,l+1 )
        u_s[string(i)] = Array{Float64}(undef, L,num_label,l+1) #zeros(L,num_label,l+1 )
        w_s[string(i)] = Array{Float64}(undef, L,num_label,l+1) #zeros(L,num_label,l+1 )
        R[string(i)] = Array{Float64}(undef, L,num_label,l+1) #zeros(L,num_label,l+1 )
        θ[string(i)] = Array{Float64}(undef,L,num_label,l+1) #zeros(L,num_label,l+1 )
    end

    local J
    u_gibbs = Array{Float64}(undef, L,2,l+1 ) #zeros(L,num_chains,l+1 )
    w_gibbs = Array{Float64}(undef, L,2,l+1 ) #zeros(L,num_chains,l+1 )
        
    for i in 0:l
        idx = seq ? i : Int(s/l*i)
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

            R[string(num_label)][:,:,i+1] = getm_std(.√ ((x[1:L,:] .- xsp) .^2 .+ (y[1:L,:] .- ysp) .^2))
            θ[string(num_label)][:,:,i+1] = getm_std(atan.((y[1:L,:] .- ysp) , (x[1:L,:] .- xsp)))
            x_s[string(num_label)][:,:,i+1] = getm_std(x)
            y_s[string(num_label)][:,:,i+1] = getm_std(y);
            u_s[string(num_label)][:,:,i+1] = getm_std(1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp))
            w_s[string(num_label)][:,:,i+1] = getm_std(1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp))
        end

        λs[:,i+1] = Array(F.S)
        a0s[:,i+1] = Array(F.U' * J.a)
        b0s[:,i+1] = Array(F.Vt * J.b)
        μ_u[:,i+1] = 1/sqrt(2) .* (μ_x[1:L] .+ μ_y[1:L] .- xsp .- ysp)
        μ_w[:,i+1] = 1/sqrt(2) .* (-μ_x[1:L] .+ μ_y[1:L] .+ xsp .- ysp)

        v,h = gibbs_sample(J, hparams, num_chains,500)
        x = cpu(F.U' * v)
        y = cpu(F.Vt * h);
        u_gibbs[:,:,i+1] = getm_std(1/sqrt(2) .* (x[1:L,:] .+ y[1:L,:] .- xsp .- ysp))
        w_gibbs[:,:,i+1] = getm_std(1/sqrt(2) .* (-x[1:L,:] .+ y[1:L,:] .+ xsp .- ysp))
    end
    return λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs
end

@time λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs = get_landscape_params(modelName, hparams, dict, 100, seq=true);
λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs = get_landscape_params(hparams, 2)
@time λs, a0s, b0s, x_s, y_s, R, θ, u_s, w_s, μ_u, μ_w, u_gibbs, w_gibbs = get_landscape_params_mstd(modelName, hparams, dict, 1000);

f_uw(u,w, a0, b0, λ) = a0[1:hparams.nh,:] .* b0 ./ λ .- λ .^2 .* (u .^ 2 .- w .^ 2) ./ 2

begin
    idx=1
    label="0"
    L = min(hparams.nh, hparams.nv)
    p1 = hline([0], ls=:dash, color=:black, label="Saddle point", 
        ylabel="Position u", title="$(idx)")
    sgn = sign.(μ_w[idx,:])
    # sgn = ones(21)
    p1 = plot!(sgn .* μ_u[idx,:], label="μ", marker=:circle, markerstrokewidth=0.0, markersize=5, lw=2)
    p1 = plot!(sgn .* u_s[label][idx,1,:], label="Test data", marker=:circle, 
        markerstrokewidth=0.0, markersize=5, lw=2, ribbon=u_s[label][idx,2,:])
    p1 = plot!(sgn .* u_gibbs[idx,1,:], label="Gibbs data", marker=:circle, 
        markerstrokewidth=0.0, markersize=2, lw=2, ribbon=u_gibbs[idx,2,:], frame=:box, 
        legend = :outertopright) #, xlim=(0,100), ylim=(-5,19))

    p2 = hline([0], ls=:dash, color=:black, label="Saddle point",
        ylabel="Position w",)
    p2 = plot!(sgn .* μ_w[idx,:], label="μ", marker=:circle, markerstrokewidth=0.0, markersize=8, lw=2)
    p2 = plot!(sgn .* w_s[label][idx,1,:], label="Test data", marker=:circle, 
        markerstrokewidth=0.0, markersize=5, lw=2, ribbon=w_s[label][idx,2,:])
    p2 = plot!(sgn .* w_gibbs[idx,1,:], label="Gibbs sampled", marker=:circle, 
        markerstrokewidth=0.0, markersize=2, lw=2, ribbon= w_gibbs[idx,2,:], frame=:box,
        legend = :outertopright) #, xlim=(30,40))

    p3 = plot(λs[idx,:], ls=:dash, color=:black, label="Sing value", ylabel="", xlabel="Epoch" )

    # p4 = hline([0], ls=:dash, color=:black, label="Saddle point", 
    #     ylabel="Position u", title="$(idx)")
    # p4 = plot!(sqrt.( ( μ_u[idx,:] ) .^ 2 .+ ( μ_w[idx,:]) .^2), label="μ", marker=:circle, markerstrokewidth=0.0, markersize=5, lw=2)
    # p4 = plot!(sqrt.( (u_s[label][idx,1,:] ) .^ 2 .+ (w_s[label][idx,1,:]) .^ 2), label="Test data", marker=:circle, 
    #     markerstrokewidth=0.0, markersize=5, lw=2, ribbon=u_s[label][idx,2,:])
    # p4 = plot!(sqrt.( (u_gibbs[idx,1,:]) .^ 2 .+ (w_gibbs[idx,1,:]) .^ 2 ), label="Gibbs data", marker=:circle, 
    #     markerstrokewidth=0.0, markersize=2, lw=2, ribbon=u_gibbs[idx,2,:], frame=:box, 
    #     legend = :outertopright)

    # p5 = hline([0], ls=:dash, color=:black, label="Saddle point", 
    #     ylabel="Position u", title="$(idx)")
    # p5 = plot!(atan.( μ_w[idx,:], μ_u[idx,:] ) , label="μ", marker=:circle, markerstrokewidth=0.0, markersize=5, lw=2)
    # p5 = plot!(atan.( w_s[label][idx,1,:], u_s[label][idx,1,:]), label="Test data", marker=:circle, 
    #     markerstrokewidth=0.0, markersize=5, lw=2, ribbon=u_s[label][idx,2,:])
    # p5 = plot!(atan.( w_gibbs[idx,1,:], u_gibbs[idx,1,:]), label="Gibbs data", marker=:circle, 
    #     markerstrokewidth=0.0, markersize=2, lw=2, ribbon=u_gibbs[idx,2,:], frame=:box, 
    #     legend = :outertopright)

    # p5 = plot(atan.( μ_w[idx,:], μ_u[idx,:] ), sqrt.( ( μ_u[idx,:] ) .^ 2 .+ ( μ_w[idx,:]) .^2), proj = :polar,
        # marker=:circle, markerstrokewidth=0.0, markersize=5, lw=2, color=[RGB(1/n,0,1-1/n) for n in 1:501])

    p = plot(p1,p2,p3, layout=(3,1), size=(600,900), left_margin=5mm)
    # savefig(p, PATH * "Symmetry_$(modelName)_$(idx).png")
    p
end

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

    p3 = plot(λs[idx,:], ls=:dash, color=:black, label="Sing value", ylabel="", xlabel="Epoch" )

    p = plot(p1,p2,p3, layout=(3,1), size=(600,900), left_margin=5mm)
    # savefig(p, PATH * "Symmetry_$(modelName)_$(idx).png")
    p
end

# idx=1
# sgn = sign.(μ_w[idx,:])
# ss = size(sgn,1)-1
# plot( sgn .* μ_w[idx,:], sgn .* μ_u[idx,:] , proj = :cartesian,
#         marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, opacity=0.5, color=[RGB(1-n/ss,0,n/ss) for n in 0:ss])
# plot!( sgn .* w_gibbs[idx,1,:], sgn .* u_gibbs[idx,1,:] , proj = :cartesian,
#     marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, opacity=0.5, color=[RGB(1-n/ss,0.5,n/ss) for n in 0:ss])
# plot!( sgn .*  w_s[label][idx,1,:], sgn .*  u_s[label][idx,1,:] , proj = :cartesian,
#     marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, opacity=0.5, color=[RGB(1-n/ss,1.0,n/ss) for n in 0:ss])



# plot(atan.( sgn .* μ_w[idx,:], sgn .* μ_u[idx,:] ), sqrt.( ( μ_u[idx,:] ) .^ 2 .+ ( μ_w[idx,:]) .^2), proj = :cartesian,
#         marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, opacity=0.5, color=[RGB(1-n/ss,0,n/ss) for n in 0:ss])
# plot!(atan.( sgn .* w_gibbs[idx,1,:], sgn .* u_gibbs[idx,1,:] ), sqrt.( ( u_gibbs[idx,1,:] ) .^ 2 .+ ( w_gibbs[idx,1,:]) .^2), proj = :cartesian,
#     marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, opacity=0.7, color=[RGB(1-n/ss,0.5,n/ss) for n in 0:ss])
# plot!(atan.( sgn .*  w_s[label][idx,1,:], sgn .*  u_s[label][idx,1,:] ), sqrt.( (  u_s[label][idx,1,:] ) .^ 2 .+ (  w_s[label][idx,1,:]) .^2), proj = :cartesian,
#     marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, opacity=0.5, color=[RGB(1-n/ss,1.0,n/ss) for n in 0:ss])

# r_μ = sqrt.( ( μ_u[idx,:] ) .^ 2 .+ ( μ_w[idx,:]) .^2)
# th_μ = atan.( sgn .* μ_w[idx,:], sgn .* μ_u[idx,:] )
# r_gibbs = sqrt.( ( u_gibbs[idx,1,:] ) .^ 2 .+ ( w_gibbs[idx,1,:]) .^2)
# th_gibbs = atan.( sgn .* w_gibbs[idx,1,:], sgn .* u_gibbs[idx,1,:] )
# r_s = sqrt.( (  u_s[label][idx,1,:] ) .^ 2 .+ (  w_s[label][idx,1,:]) .^2)
# th_s = atan.( sgn .*  w_s[label][idx,1,:], sgn .*  u_s[label][idx,1,:] )

# plot(r_μ[3:ss+1],r_μ[3:ss+1] .- r_μ[2:ss], marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, 
#     opacity=0.5, color=[RGB(1-n/(ss-1),0,n/(ss-1)) for n in 0:(ss-2)])
# plot!(r_gibbs[3:ss+1],r_gibbs[3:ss+1] .- r_gibbs[2:ss], marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, 
#     opacity=0.5, color=[RGB(1-n/499,0.5,n/499) for n in 0:498])
# plot!(r_s[3:ss+1],r_s[3:ss+1] .- r_s[2:ss], marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, 
#     opacity=0.5, color=[RGB(1-n/(ss-1),1.,n/(ss-1)) for n in 0:(ss-1)])

# plot(th_μ[3:ss+1],th_μ[3:ss+1] .- th_μ[2:ss], marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, 
#     opacity=0.5, color=[RGB(1-n/(ss-1),0,n/(ss-1)) for n in 0:(ss-2)])
# plot!(th_gibbs[3:ss+1],th_gibbs[3:ss+1] .- th_gibbs[2:ss], marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, 
#     opacity=0.5, color=[RGB(1-n/(ss-1),0.5,n/(ss-1)) for n in 0:(ss-2)])
# plot!(th_s[3:ss+1],th_s[3:ss+1] .- th_s[2:ss], marker=:circle, markerstrokewidth=0.1, markersize=7, lw=2, 
#     opacity=0.5, color=[RGB(1-n/(ss-1),1.,n/(ss-1)) for n in 0:(ss-2)])
##################
#########cholesky
# rbm, J, m, hparams, rbmZ = initModel(nv=hparams.nv, nh=hparams.nh, batch_size=500, lr=1.5, t=10, gpu_usage = true, optType="Adam")
ep = 70
J = load("$(dict["bdir"])/models/$(modelName)/J/J_$(ep).jld", "J")
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
plot(heatmap(cpu(mat_rot)), title=ep)


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