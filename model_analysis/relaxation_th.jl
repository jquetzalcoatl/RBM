using LinearAlgebra, Plots, OMEinsum, Plots.PlotMeasures
# include("../utils/train.jl")
include("../utils/init.jl")
include("../utils/en.jl")
include("../scripts/exact_partition.jl")
include("../scripts/PhaseAnalysis.jl")

function generate_corr_plots(J::Weights, hparams::HyperParams, energy_state::Bool=false)
    # Generate reference data
    @info "Generate reference data"
    zs = randn((hparams.nv+hparams.nh),Int(2^(hparams.nv+hparams.nh))) * 0.5
    Δ = (zs .- mean(zs,dims=2))
    @ein cor2body_ref[i,j] := Δ[i,k] * Δ[j,k]
    cor2body_ref = cor2body_ref * 2.0^(-(hparams.nv+hparams.nh))
    idx_2 = findall(x->x>0.2, cor2body_ref)
    @info idx_2

    @ein cor3body_ref[i,j,l] := Δ[i,k] * Δ[j,k] * Δ[l,k]
    cor3body_ref = cor3body_ref * 2.0^(-(hparams.nv+hparams.nh))

    @ein cor4body_ref[i,j,l,m] := Δ[i,k] * Δ[j,k] * Δ[l,k] * Δ[m,k]
    cor4body_ref = cor4body_ref * 2.0^(-(hparams.nv+hparams.nh))
    idx_4a = findall(x->x>2*0.25^2-0.001 , cor4body_ref)
    idx_4b = findall(x->x<2*0.25^2-0.001 && x>0.25^2-0.001, cor4body_ref)
    @info idx_4a
    @info idx_4b

    # Generate our data
    @info "Generate our data"
    F = LinearAlgebra.svd(J.w, full=true)
    rbm_vectors = generate_binary_states(hparams.nv + hparams.nh)
    vs = hcat(rbm_vectors...)[1:hparams.nv,:]
    hs = hcat(rbm_vectors...)[hparams.nv+1:end,:]
    if energy_state
        rbm.v = vs
        rbm.h = hs
        energy_exact = H(rbm, J)
        v_gibbs, h_gibbs = gibbs_sample(J, hparams)
        rbm.v = v_gibbs
        rbm.h = h_gibbs
        energy_gibbs = H(rbm, J)
    else
        energy_exact = 0
        energy_gibbs = 0
    end
    ys = F.Vt * hs
    xs = F.U' * vs
    a = F.U' * J.a
    b = F.Vt * J.b
    x0 = -b ./ F.S
    y0 = -a[1:hparams.nh] ./ F.S

    us = 1/sqrt(2) .* (xs[1:hparams.nh,:] .- ys .- x0 .+ y0)
    ws = 1/sqrt(2) .* (xs[1:hparams.nh,:] .+ ys .- x0 .- y0)

    zs = vcat(us,ws,xs[hparams.nh+1:end,:])
    
    Δ = (zs .- mean(zs,dims=2))

    @ein cor2body[i,j] := Δ[i,k] * Δ[j,k]
    cor2body = cor2body * 2.0^(-(hparams.nv+hparams.nh))

    #three body
    @ein cor3body[i,j,l] := Δ[i,k] * Δ[j,k] * Δ[l,k]
    cor3body = cor3body * 2.0^(-(hparams.nv+hparams.nh))
    # heatmap(cat([cat([cor3body[:,:,i+4*j] for i in 1:4]...,dims=2) for j in 0:3]...,dims=1))

    #four body
    @ein cor4body[i,j,l,m] := Δ[i,k] * Δ[j,k] * Δ[l,k] * Δ[m,k]
    cor4body = cor4body * 2.0^(-(hparams.nv+hparams.nh))

    f1 = plot(cor4body[idx_4a] ./ (3*0.25^2) .- 1 , lw=0.8, 
        markersize=5, markershape=:circle, 
        markerstrokewidth=0, label="RBM", 
        ylabel="σ₄ rel deviation", xlabel="Index",
        frame=:box, legendfontsize=15, tickfontsize=15, 
        labelfontsize=15, size=(700,500))
    f1 = plot!(cor4body_ref[idx_4a] ./ (3*0.25^2) .- 1 , lw=0.8, 
        markersize=5, markershape=:circle, 
        markerstrokewidth=0, label="Normal dist data")

    f2 = plot(cor4body[idx_4b] ./ (0.25^2) .- 1 , lw=0.8, 
        markersize=5, markershape=:circle, 
        markerstrokewidth=0, label="RBM", 
        ylabel="σ₂σ₂ rel deviation", xlabel="Index",
        frame=:box, legendfontsize=15, tickfontsize=15, 
        labelfontsize=15, size=(700,500))
    f2 = plot!(cor4body_ref[idx_4b] ./ (0.25^2) .- 1 , lw=0.8, 
        markersize=5, markershape=:circle, 
        markerstrokewidth=0, label="Normal dist data")

    f3 = plot(cor2body[idx_2] ./ 0.25 .- 1, lw=0.5, 
        markersize=5, markershape=:circle, 
        markerstrokewidth=0, label="RBM",
        ylabel="σ₄ rel deviation", xlabel="Index",
        frame=:box, legendfontsize=15, tickfontsize=15, 
        labelfontsize=15, size=(700,500))
    f3 = plot!(cor2body_ref[idx_2] ./ (0.25) .- 1 , lw=0.8, 
        markersize=5, markershape=:circle, 
        markerstrokewidth=0, label="Normal dist data")

    f4 = plot(reshape(cor3body,:), lw=0.5, 
        markersize=5, markershape=:circle, 
        markerstrokewidth=0, label="RBM",
        ylabel="3-point corr", xlabel="Index",
        frame=:box, legendfontsize=15, tickfontsize=15, 
        labelfontsize=15, size=(700,500))
    f4 = plot!(reshape(cor3body_ref,:), lw=0.8, 
        markersize=5, markershape=:circle, 
        markerstrokewidth=0, opacity=0.2, label="Normal dist data")
    fig = plot(f1,f2,f3,f4, layout=(2,2), size=(1000,800), left_margin=5mm)
    fig, energy_exact, energy_gibbs
end

PATH = "/home/javier/Projects/RBM/NewResults/PhaseDiagrams/"

rbm, J, m, hparams, rbmZ = initModel(nv=10, nh=6, batch_size=500, lr=1.5, t=10, gpu_usage = false, optType="Adam")

####################
subWeights = J.w[1:10,1:6]
suba = J.a[1:10]
subb = J.b[1:6]

J.w = subWeights |> cpu
J.a = suba |> cpu
J.b = subb |> cpu
######################

J.w = rand(size(J.w)[1], size(J.w)[2])
J.a = rand(size(J.a)[1])
J.b = rand(size(J.b)[1])

ff, energy_exact, energy_gibbs = generate_corr_plots(J, hparams, true)
ff
savefig(ff, PATH * "fig_nv10_nh6_Submodel.png")

fig = plot(energy_exact, st=:histogram, normalized=true, lw=0.0, 
    bins=200, tickfontsize=15, frame=:box, label="Exact", 
    legendfontsize=15)
fig = plot!(energy_gibbs, st=:histogram, normalized=true, lw=0, 
    opacity=0.5, bins=100, label="Gibbs samples")
savefig(fig, PATH * "histogram_subModel_weights_and_bias_2.png")
##########################################################
# begin
#     # include("../therm.jl")
#     include("../configs/yaml_loader.jl")
#     PATH = "/home/javier/Projects/RBM/Results/"
#     # dev = gpu
#     # β = 1.0
#     config, _ = load_yaml_iter();
# end
config.model_analysis["files"]

modelName = "PCD-FMNIST-500-replica1-L" #config.model_analysis["files"][1]
# modelName = "CD-FMNIST-500-T1000-BW-replica1-L"
modelName = config.model_analysis["files"][1]
modelName = "Random-RBM_small"
rbm, J, m, hparams, opt = loadModel(modelName, gpu, idx=100);
################################################### save plots