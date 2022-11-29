using LinearAlgebra, FFTW

mutable struct ObsMagnetization{NDIMS, DIMS, QCLOCK, cis_q} # NDIMS is needed!
  c_measure::Int64

  #M_ET::Array{Complex{Float64}, 1}
  M2_ET::Array{Float64, 1}
  M4_ET::Array{Float64, 1}
  energy::Array{Float64, 1}

  chi_local::Array{Float64, 1} # local spin-spin susceptibility
  chi_global::Array{Float64, 1}

  cos_q_theta::Array{Float64, 1} #U(1)-Zq order parameter

  abs2_S_q_ET::Array{Array{Float64, 1}, NDIMS}
  G_r::Array{Array{Complex{Float64}, 1}, NDIMS}

  q_config::Array{microint, NDIMS}
  cis_q_config::Array{Complex{Float64}, NDIMS}
  tot_cis_q_config::Array{Complex{Float64}, NDIMS}

  fft_cis_q_config::Array{Complex{Float64}, NDIMS}
  tot_abs2_S_q_ET::Array{Float64, NDIMS}
  g_r::Array{Complex{Float64}, NDIMS}

  fft_plan::FFTW.cFFTWPlan{Complex{Float64}, -1, false, NDIMS, NTuple{NDIMS, Int64}}

  # constructor
  function ObsMagnetization(DIMS::NTuple{NDIMS, Int64}, cis_q::NTuple{QCLOCK, Complex{Float64}}, num_of_measure::Int64, beta::Float64) where {NDIMS, QCLOCK}
    cis_q_config = zeros(Complex{Float64}, DIMS)

    new{NDIMS, DIMS, QCLOCK, cis_q}(
        1,

        #zeros(Float64, num_of_measure),
        zeros(Float64, num_of_measure),
        zeros(Float64, num_of_measure),
        zeros(Float64, num_of_measure),

        zeros(Float64, num_of_measure),
        zeros(Float64, num_of_measure),

        zeros(Float64, num_of_measure),

        Array{Float64, 1}[zeros(Float64, num_of_measure) for site in CartesianIndices(DIMS)],
        Array{Float64, 1}[zeros(Float64, num_of_measure) for site in CartesianIndices(DIMS)],

        zeros(microint, DIMS),
        cis_q_config,
        zeros(Complex{Float64}, DIMS),

        zeros(Complex{Float64}, DIMS),
        zeros(Float64, DIMS),
        zeros(Float64, DIMS),

        plan_fft(cis_q_config, Tuple(1:NDIMS))
      )
  end
end

function measure_all!(coupling_config::CouplingConfig{NDIMS, DIMS, QCLOCK}, sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}, obs::ObsMagnetization{NDIMS, DIMS, QCLOCK, cis_q}) where {NDIMS, DIMS, QCLOCK, cis_q}

  tot_m_ET = 0.0im

  for site in CartesianIndices(DIMS)
    state = sse_config.q_config[site]

    obs.q_config[site] = state
    obs.cis_q_config[site] = cis_q[state]
    tot_m_ET += cis_q[state]
  end

  M_ET = 0.0
  M2_ET = 0.0
  M4_ET = 0.0

  fill!(obs.tot_abs2_S_q_ET, 0.0)
  fill!(obs.tot_cis_q_config, 0.0)
  mul!(obs.fft_cis_q_config, obs.fft_plan, obs.cis_q_config)

  for (p, op) in enumerate(sse_config.op_array)
    if op.op_type != id_op
      if op.op_type == field && op.colour[1]!=op.colour[2] # if op is a spin-flip field term
        state = obs.q_config[CartesianIndex(op.site)] # old state

        tot_m_ET += - cis_q[state]

        state = op.colour[2] # new state

        obs.q_config[CartesianIndex(op.site)] = state
        obs.cis_q_config[CartesianIndex(op.site)] = cis_q[state]
        tot_m_ET += cis_q[state]

        mul!(obs.fft_cis_q_config, obs.fft_plan, obs.cis_q_config)
      end

      M_ET += tot_m_ET

      M2_ET += abs2(tot_m_ET)
      M4_ET += abs2(tot_m_ET)^2
      obs.tot_abs2_S_q_ET .+= abs2.(obs.fft_cis_q_config)
      obs.tot_cis_q_config .+= obs.cis_q_config
    end
  end

  len = sse_config.num_of_op
  obs.chi_global[obs.c_measure]=abs2(M_ET)+M2_ET
  obs.chi_global[obs.c_measure] *= coupling_config.beta/(len*(len+1))
  obs.M2_ET[obs.c_measure] = M2_ET/len
  obs.M4_ET[obs.c_measure] = M4_ET/len
  obs.cos_q_theta[obs.c_measure]=cos(QCLOCK*angle(M_ET))
  obs.energy[obs.c_measure] = -len/coupling_config.beta
  obs.g_r=ifft(obs.tot_abs2_S_q_ET/(len*prod(DIMS)))
  for site in CartesianIndices(DIMS)
    obs.abs2_S_q_ET[site][obs.c_measure] = obs.tot_abs2_S_q_ET[site]/len
    obs.chi_local[obs.c_measure] += abs2(obs.tot_cis_q_config[site])+len
    obs.G_r[site][obs.c_measure]=obs.g_r[site]
  end

  obs.chi_local[obs.c_measure] *= coupling_config.beta/(len * (len+1) * prod(DIMS))

  obs.c_measure+=1
  #obs.c_measure = mod1(obs.c_measure+1,length(obs.M2_ET))
end
