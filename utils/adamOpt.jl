raw"""
This is taken from Adamopt.jl and slightly adapted.
https://gist.github.com/vankesteren/96207abcd16ecd01a2491bcbec12c73f
"""

mutable struct Adam
  theta # Parameter array
#   loss::Function                # Loss function
#   grad                          # Gradient function
  m #::AbstractArray{Float64}     # First moment
  v #::AbstractArray{Float64}     # Second moment
  b1 #::Float64                   # Exp. decay first moment
  b2 #::Float64                   # Exp. decay second moment
  a #::Float64                    # Step size
  eps #::Float64                  # Epsilon for stability
  t #::Int                        # Time step (iteration)
end

# Outer constructor
# function Adam(theta::AbstractArray{Float64}, loss::Function, grad::Function)
function Adam(theta, a; dev)
  m   = zeros(size(theta)) |> dev
  v   = zeros(size(theta)) |> dev
  b1  = 0.0
  b2  = 0.999
#   a   = 0.001
  eps = 1e-6
  t   = 0
  Adam(theta, m, v, b1, b2, a, eps, t)
end

# Step function with optional keyword arguments for the data passed to grad()
function step!(opt::Adam, Δ)
  opt.t += 1
  gt    = Δ # opt.grad(opt.theta; data...)  #<---- Delta W
  opt.m = opt.b1 .* opt.m + (1 - opt.b1) .* gt
  opt.v = opt.b2 .* opt.v + (1 - opt.b2) .* gt .^ 2
  mhat = opt.m ./ (1 - opt.b1^opt.t)
  vhat = opt.v ./ (1 - opt.b2^opt.t)
  opt.theta = opt.theta + opt.a .* (mhat ./ (sqrt.(vhat) .+ opt.eps))
  opt.theta
end