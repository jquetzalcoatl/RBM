using CUDA, Flux, JLD2
using Plots.PlotMeasures

include("utils/train.jl")

function loadLandscapes(x_i, y_i, PATH = "/home/javier/Projects/RBM/Results/",  modelname = "CD-500-T1000-5-BW-replica1-L"; l=30, nv=28*28, nh=500)
    s = size(readdir("$(PATH)/models/$(modelname)/J"),1)
    a0s = Array(zeros(nv,l))
    b0s = Array(zeros(nh,l))
    λs = Array(zeros(nh,l))

    x_s = Dict()
    y_s = Dict()
    R = Dict()
    Θ = Dict()

    Δidx = s >= l ? Int(floor(s/l)) : 1
    for i in 1:min(l,s)
        idx = Δidx*i
        
        J = load("$(PATH)/models/$(modelname)/J/J_$(idx).jld", "J")
        J.w = gpu(J.w)
        J.b = gpu(J.b)
        J.a = gpu(J.a)
        F = LinearAlgebra.svd(J.w, full=true);

        for num_label in 0:9
            v = gpu(x_i[:,y_i .== num_label])
            # h = Array{Float32}(sign.(rand(500, size(v,2)) |> dev .< σ.(β .* (dev(J.w)' * v .+ dev(J.b))))) |> dev
            h = Array{Float32}(σ.(β .* (dev(J.w)' * v .+ dev(J.b)))) |> dev

            x = cpu(F.U' * v)
            y = cpu(F.Vt * h);


            xsp = -Array(F.Vt * J.b) ./ Array(F.S)
            ysp = -Array(F.U' * J.a)[1:500] ./ Array(F.S);

            if i != 1
                R[string(num_label)] = cat(R[string(num_label)], .√ ((x[1:500,:] .- xsp) .^2 .+ (y .- ysp) .^2), dims=3)
                Θ[string(num_label)] = cat(Θ[string(num_label)], atan.((y .- ysp) , (x[1:500,:] .- xsp)), dims=3)
                x_s[string(num_label)] = cat(x_s[string(num_label)], x, dims=3)
                y_s[string(num_label)] = cat(y_s[string(num_label)], y, dims=3);
            else
                R[string(num_label)] = .√ ((x[1:500,:] .- xsp) .^2 .+ (y .- ysp) .^2)
                Θ[string(num_label)] = atan.((y .- ysp) , (x[1:500,:] .- xsp))
                x_s[string(num_label)] = x
                y_s[string(num_label)] = y;
            end
        end

        λs[:,i] = Array(F.S)
        a0s[:,i] = Array(F.U' * J.a)
        b0s[:,i] = Array(F.Vt * J.b)

    end
    genAnimation(x_s, y_s, R, Θ, s, λs, a0s, b0s, modelname)
    return λs, a0s, b0s, R, Θ, x_s, y_s
end

function saveModePlot(λs, sp, a0s, b0s, R, Θ, x_s, y_s, modelname)
    isdir("$(PATH)/Figs/$(modelname)") || mkpath("$(PATH)/Figs/$(modelname)")
    f = plot();
    L = 500
    for i in 1:L
        f = plot!(λs[i,:], lw=2.5, c=RGB(i/L,0,1-i/L));
    end
    f = plot!(yscale=:log, size=(700,500), xlabel="Epochs (x10)", ylabel="λ modes", frame=:box, legend=:none, margin = 15mm);
    savefig(f, "$(PATH)/Figs/$(modelname)/modes_$(modelname).png")
    
    f = plot();
    for i in 1:L
        f = plot!(sp[i,:], lw=2.5, c=RGB(1/i,0,1-1/i), opacity=1/(0.2*i));
    end
    f = plot!(xlabel="Epochs (x10)", ylabel="Saddle Point", frame=:box, legend=:none, margin = 15mm);
    savefig(f, "$(PATH)/Figs/$(modelname)/saddle_points_$(modelname).png")
    
    f = plot(reshape(sum(sp, dims=1),:), xlabel="Epochs (x10)", ylabel="Energy at Saddle Points", frame=:box, legend=:none, lw=2.5, margin = 15mm);
    savefig(f, "$(PATH)/Figs/$(modelname)/energy_saddle_points_$(modelname).png")
    
    jldsave("$(PATH)/Figs/$(modelname)/landscapeParameters.jld", sp=sp, λs=λs, xsp=-b0s ./ λs, ysp= -a0s ./ λs)
    jldsave("$(PATH)/Figs/$(modelname)/validationRS.jld", R=R, Θ=Θ, x_s=x_s, y_s=y_s)
    
    
    # save("$(PATH)/Figs/$(modelname)/sp.jld", "sp", sp)
    # save("$(PATH)/Figs/$(modelname)/ls.jld", "ls", λs)
    # save("$(PATH)/Figs/$(modelname)/xsp.jld", "xsp", -b0s ./ λs)
    # save("$(PATH)/Figs/$(modelname)/ysp.jld", "ysp", -a0s ./ λs)
end

f(x,y,i, a0, b0, λ) = - (a0[i]*x + b0[i]*y + λ[i]*x*y )

function genAnimation(x_s, y_s, R, Θ, s, λs, a0s, b0s, modelname)
    isdir("$(PATH)/Figs/$(modelname)/anim") || mkpath("$(PATH)/Figs/$(modelname)/anim")
    for num_label in string.(collect(0:9))
        anim = @animate for ep ∈ 1:size(R[num_label],3)
            f = plot();
            for i in 1:15
                f = plot!(Θ[num_label][i,:,ep], R[num_label][i,:,ep], proj = :polar, m = 4, markerstrokewidth=0.1, 
                    title="Label:$(num_label) \n Epoch: $(ep*s/size(R[num_label],3))", label="EL $i");
            end
            f = plot!(size=(550,550), margin = 15mm);
            if ep==1 || ep== size(R[num_label],3)
                savefig(f, "$(PATH)/Figs/$(modelname)/anim/polar_plot_$(num_label)_$(ep)_$(modelname).png")
            end
        end
        gif(anim, "$(PATH)/Figs/$(modelname)/anim/anim_polar_$(num_label).gif", fps = 1)
    end
    
    for enLandIdx in 1:10
        anim = @animate for ep ∈ 1:size(x_s["0"],3)
            p = plot(-15:10, -10:13, (x,y)->f(x,y,enLandIdx, a0s[:,ep], b0s[:,ep], λs[:,ep]), 
                st=:contourf, c=cgrad(:matter, 105, rev=true, scale = :exp, categorical=false), xlabel="x", ylabel="y", clabels=true);
            for num_label in string.(collect(0:9))
                plot!(x_s[num_label][enLandIdx,:,ep], y_s[num_label][enLandIdx,:,ep], st=:scatter, markerstrokewidth=0.1, 
                title="En Landscaped:$(enLandIdx) \n Epoch: $(ep*s/size(x_s[num_label],3))");
            end
            plot!([-b0s[enLandIdx,ep]/λs[enLandIdx,ep]],[-a0s[enLandIdx,ep]/λs[enLandIdx,ep]], ms=15, st=:scatter, c=:black, legend=:none, markershape=:x);
            plot!(size=(550,550), margin = 15mm);
            if ep==1 || ep== size(x_s["0"],3)
                savefig("$(PATH)/Figs/$(modelname)/anim/landscape_plot_$(enLandIdx)_$(ep)_$(modelname).png")
            end
        end
        gif(anim, "$(PATH)/Figs/$(modelname)/anim/anim_LS_$(enLandIdx).gif", fps = 1)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    PATH = "/home/javier/Projects/RBM/Results/"
    l=100
    # nv=28*28
    # nh=500
    dev = gpu
    β = 1.0
    # modelName = "CD-500-T1000-5-BW-replica1-L"
    # rbm, J, m, hparams, opt = loadModel(modelName, gpu);
    # x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
    # for model in ["Rdm-500-T10-BW-replica", "Rdm-500-T100-BW-replica", "CD-500-T1-replica", "CD-500-T1-BW-replica", "CD-500-T10-BW-replica", "CD-500-T100-BW-replica"]
    # for model in ["CD-500-T1-replica", "CD-500-T1-BW-replica", "CD-500-T10-BW-replica", "CD-500-T100-BW-replica"]
    for model in ["PCD-500-1200-replica", "PCD-500-784-replica"]
        modelName = model * "1"
        rbm, J, m, hparams, opt = loadModel(modelName, gpu);
        x_i, y_i = loadData(; hparams, dsName="MNIST01", numbers=collect(0:9), testset=true);
        for i in 1:5

            modelname = model * "$(i)"
            @info modelname
            λs, a0s, b0s, R, Θ, x_s, y_s = loadLandscapes(x_i, y_i, PATH, modelname; l, hparams.nv, hparams.nh);
            sp = a0s[1:nh,:] .* b0s ./ λs;
            saveModePlot(λs, sp, a0s[1:nh,:], b0s, R, Θ, x_s, y_s, modelname)
        end
    end
end