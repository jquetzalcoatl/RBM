function gibbs_sample(J::Weights, hparams::HyperParams, dev, num::Int=5000, steps::Int=1000)
    v = rand([0,1],hparams.nv, num) |> dev
    local h
    for _ in 1:steps
        h = sign.(rand(hparams.nh, num) |> dev .< σ.(J.w' * v .+ J.b))
        v = sign.(rand(hparams.nv, num) |> dev .< σ.(J.w * h .+ J.a))
    end
    v, h
end

function gibbs_sample(v, J::Weights, hparams::HyperParams, dev, num::Int=5000, steps::Int=1000)
    v = v |> dev
    local h
    for _ in 1:steps
        h = sign.(rand(hparams.nh, num) |> dev .< σ.(J.w' * v .+ J.b))
        v = sign.(rand(hparams.nv, num) |> dev .< σ.(J.w * h .+ J.a))
    end
    v, h
end

# function gibbs_sampling(v,h,J; mcs=5000, dev=gpu, β=1)
#     nh = size(h,1)
#     nv = size(v,1)
#     num= minimum([size(v,2), 1000])
#     for i in 1:mcs
#         h = Array{Float32}(sign.(rand(nh, num) |> dev .< σ.( (β .* J.w)' * v .+ J.b))) |> dev
#         v = Array{Float32}(sign.(rand(nv, num) |> dev .< σ.( (β .* J.w) * h .+ J.a))) |> dev 
#     end
#     return v,h
# end

function AIS(J::Weights, hparams::HyperParams, samples::Int=500, mcs::Int=5000, nbeta::Float64=30, dev=gpu)
    lnZa = sum(log1p.(exp.(J.a))) + sum(log1p.(exp.(J.b)))
    FreeEnergy_ratios = 0.0
    Δbeta = 1.0 / nbeta

    # v,h = rand([0,1],hparams.nv,samples) |> dev,rand([0,1],hparams.nh,samples) |> dev
    v = rand([0,1],hparams.nv,samples)
    for β in 0:Δbeta:1.0-Δbeta
        @info "AIS annealing $β"
        v,h = gibbs_sample(v,J,hparams, dev,samples, mcs)

        energy_samples_i = energy_samples(v,h,J,β)
        energy_samples_i_plus = energy_samples(v,h,J,β + Δbeta)
        FreeEnergy_ratios += log(mean(exp.(energy_samples_i .- energy_samples_i_plus)))
    end
    logZb = FreeEnergy_ratios + lnZa
    return logZb
end

function RAIS(J::Weights, hparams::HyperParams, samples::Int=500, mcs::Int=5000, nbeta::Float64=30, dev=gpu)
    lnZb = sum(log1p.(exp.(J.a))) + sum(log1p.(exp.(J.b)))
    FreeEnergy_ratios = 0.0
    Δbeta = 1.0 / nbeta

    # v,h = rand([0,1],hparams.nv,samples) |> dev,rand([0,1],hparams.nh,samples) |> dev
    v = rand([0,1],hparams.nv,samples)
    for β in 1:-Δbeta:Δbeta
        @info "RAIS annealing $β"
        v,h = gibbs_sample(v,J,hparams, dev,samples, mcs)

        energy_samples_i = energy_samples(v,h,J,β)
        energy_samples_i_minus = energy_samples(v,h,J,β - Δbeta)
        FreeEnergy_ratios += log(mean(exp.(energy_samples_i .- energy_samples_i_minus)))
    end
    logZa = - FreeEnergy_ratios + lnZb
    return logZa
end

energy_samples(v,h,J::Weights,β::Float64) = - (v' * J.a + (J.b' * h)' + diag(v' * (β .* J.w) * h))
