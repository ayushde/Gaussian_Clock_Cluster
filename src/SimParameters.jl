mutable struct SimParams{NDIMS}
  L::Array{Int64, 1}
  QCLOCK::Int64
  J::Array{Array{Float64, NDIMS}, 1} # [J_x, J_y, ..., J_z]
  h::Array{Float64, NDIMS}

  max_log2_beta::Int64

  num_of_thermal::Int64
  num_of_sweep::Int64
  num_of_measure::Int64
  num_of_node_moves::Int64
  num_of_link_moves::Int64
  num_of_PT_sweeps::Int64
  num_of_moves_between_PT::Int64
  g::Float64
  # unintizlied constructor
  SimParams(NDIMS) = new{NDIMS}()
end

# Constructor from dictionary
function SimParams(sim_dict::Dict{String,Any})
  sim_params = SimParams(length(sim_dict["L"]))

  for f=fieldnames(typeof(sim_params))
    in("$f", keys(sim_dict)) ? setfield!(sim_params, Symbol(f), sim_dict["$f"]) : throw(ArgumentError("""key "$f" missing!!"""))
  end

  if size(sim_dict["h"]) != (sim_dict["L"]...,)
    throw(AssertionError("h has wrong size!!"))
  end

  if length(sim_dict["J"]) != length(sim_dict["L"])
    throw(AssertionError("J has wrong length!!"))
  else
    for i=1:length(sim_dict["L"])
      if size(sim_dict["J"][i]) != (sim_dict["L"]...,)
        throw(AssertionError("""J["$i"] has wrong size!!"""))
      end
    end
  end
  return sim_params
end
