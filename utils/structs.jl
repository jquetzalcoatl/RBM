using Parameters: @with_kw, @unpack

@with_kw struct HyperParams
    nv::Int = 28*28
    nh::Int = 100
    batch_size::Int=25
    lr::Float64 = 0.0002
    t::Int = 10
    gpu_usage::Bool = false
    optType::String = "SGD"
end

mutable struct RBM
    v
    h
end

mutable struct Weights
    w
    a
    b
end

mutable struct ModelStats
    enList
    enSDList
    ΔwList 
    ΔaList 
    ΔbList
    ΔwSDList 
    ΔaSDList 
    ΔbSDList
end

mutable struct WeightOpt
    w
    a
    b
end