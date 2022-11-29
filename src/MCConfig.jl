using GJRand

mutable struct CouplingConfig{NDIMS, DIMS, QCLOCK}
  J::Array{Float64, 1}
  h::Array{Float64, 1}

  g::Float64
  beta::Float64
  tot_coupling_const::Float64

  Q::Array{Float64,1}
  U::Array{miniint,1}

  function CouplingConfig(DIMS::NTuple{NDIMS, Int64}, QCLOCK::Int64,J::Array{Array{Float64, NDIMS}, 1}, h::Array{Float64, NDIMS}, beta::Float64, g::Float64) where NDIMS
    J_flatten = zeros(Float64, prod(DIMS)*NDIMS)
    h_flatten = zeros(Float64, prod(DIMS))
    for site in CartesianIndices(DIMS)
      ind = LinearIndices(DIMS)[site]
      h_flatten[ind] = h[site]

      for dir=1:NDIMS
        ind = LinearIndices((DIMS..., NDIMS))[Tuple(site)..., dir]
        J_flatten[ind] = J[dir][site]
      end
    end
    tot_coupling_const = (sum(h_flatten)/(sqrt(QCLOCK)) + sum(J_flatten))
    p_op = (vcat(h_flatten/(sqrt(QCLOCK)), J_flatten)/tot_coupling_const) # usual weight of each operator
    U,Q=alias_setup(p_op)

    new{NDIMS, DIMS, QCLOCK}(
      J_flatten,
      h_flatten,

      g,
      beta,
      tot_coupling_const,

      Q,
      U
    )
  end
end

mutable struct SSEConfig{NDIMS,  DIMS, QCLOCK}
  rng::GJRandRNG

  q_config::Array{microint, NDIMS}

  op_array::Array{Operator{NDIMS}, 1}
  leg_array::Array{macroint, 2}

  num_of_op::Int64 # number of non-identity operators
  num_of_ising_op::Int64 #number of bond operators

  first_spin_leg::Array{macroint, NDIMS}
  last_spin_leg::Array{macroint, NDIMS}

  cluster_flip_axis::Int64

  cluster_leg_array::Array{macroint, 1}

  node_visited_array::Array{microint,2}

  no_of_node_moves::Int64
  no_of_link_moves::Int64

  weights::NTuple{QCLOCK,Float64}

  function SSEConfig(DIMS::NTuple{NDIMS, Int64}, QCLOCK::Int64,num_of_op::Int64,num_of_ising_op::Int64, op_array::Array{Operator{NDIMS}, 1}, no_of_node_moves::Int64, no_of_link_moves::Int64) where NDIMS
    rng = GJRandRNG() # create a RNG with seed generated using entropy from the system
    coeffs=gaussian_coeffs(QCLOCK)
    op_size = length(op_array)

    new{NDIMS,  DIMS, QCLOCK}(
      rng,
      ones(microint, DIMS),
      op_array,
      fill(zero(macroint), (op_size, 4)),

      num_of_op,
      num_of_ising_op,

      fill(zero(macroint), DIMS),
      fill(zero(macroint), DIMS),

      0,

      macroint[],

      fill(zero(microint),(op_size, 2)),

      no_of_node_moves,
      no_of_link_moves,

      (coeffs...,)
    )
  end
end
