using Statistics
using Bootstrap
using ThreadTools


mutable struct Sim{NDIMS, DIMS, QCLOCK, cis_q}
  sim_params::Array{SimParams{NDIMS},1}

  coupling_configs::Array{CouplingConfig{NDIMS, DIMS, QCLOCK},1}
  sse_configs::Array{SSEConfig{NDIMS, DIMS, QCLOCK},1}
  obs::Array{ObsMagnetization{NDIMS, DIMS, QCLOCK, cis_q},1}

  coup_to_sse_map::Array{microint,1}

  function Sim(sim_params::Array{SimParams{NDIMS},1}, beta::Float64, num_of_ops::Array{Int64,1}, num_of_ising_ops::Array{Int64,1}, op_arrays::Array{Array{Operator{NDIMS}, 1}}) where {NDIMS}
    DIMS = (sim_params[1].L...,)
    QCLOCK = sim_params[1].QCLOCK
    cis_q = ([cis(2*pi*(q-1)/QCLOCK) for q=1:QCLOCK]...,)

    coupling_configs = [CouplingConfig(DIMS, QCLOCK, sim_param.J, sim_param.h, beta, sim_param.g) for sim_param in sim_params]
    sse_configs = [SSEConfig(DIMS, QCLOCK, num_of_op,num_of_ising_ops[i],op_arrays[i],sim_params[i].num_of_node_moves,sim_params[i].num_of_link_moves) for (i,num_of_op) in enumerate(num_of_ops)]
    obs = [ObsMagnetization(DIMS, cis_q, sim_param.num_of_measure, beta) for sim_param in sim_params]

    new{NDIMS, DIMS, QCLOCK, cis_q}(
        sim_params,

        coupling_configs,
        sse_configs,
        obs,

        microint.(1:length(num_of_ops))
    )
  end
end

function PT_swap!(sim::Sim)
  for i in 1:sim.sim_params[1].num_of_PT_sweeps
    coup1=rand(sim.sse_configs[1].rng,1:length(sim.sse_configs))
    coup2=coup1+rand(sim.sse_configs[1].rng,(1,-1))
    if coup2>length(sim.sse_configs) || coup2<1
      continue
    end
    J1=sim.sim_params[coup1].g
    J2=sim.sim_params[coup2].g
    sse1=sim.coup_to_sse_map[coup1]
    sse2=sim.coup_to_sse_map[coup2]
    n1=sim.sse_configs[sse1].num_of_ising_op
    n2=sim.sse_configs[sse2].num_of_ising_op
    prob_swap=(J1/J2)^(n2-n1)
    if rand(sim.sse_configs[1].rng) < prob_swap
      sim.coup_to_sse_map[coup1]=sse2
      sim.coup_to_sse_map[coup2]=sse1
    end
  end
end

function parallel_move!(sim::Sim)
  Threads.@threads for coup in 1:length(sim.coupling_configs)
  #for coup in 1:length(sim.coupling_configs)
    for i in 1:sim.sim_params[coup].num_of_moves_between_PT
      sse=sim.coup_to_sse_map[coup]
      diag_update!(sim.coupling_configs[coup],sim.sse_configs[sse])
      construct_vertex_list!(sim.sse_configs[sse])
      single_spin_update!(sim.sse_configs[sse])
      Wolff_cluster_update!(sim.sse_configs[sse])
      update_sites!(sim.sse_configs[sse])
      if DEBUG == true
        check_vertex_boundary(sim.sse_configs[sse])
        check_first_last(sim.sse_configs[sse])
        check_config(sim.sse_configs[sse].q_config,sim.sse_configs[sse].op_array,sim.sse_configs[sse])
      end
    end
  end
end

function next_move!(sim::Sim)
  parallel_move!(sim)
  PT_swap!(sim)
end

function parallel_increase_op_leg_arrays!(sim::Sim)
  Threads.@threads for sse in 1:length(sim.sse_configs)
  #for sse in 1:length(sim.sse_configs)
    increase_op_leg_array!(sim.sse_configs[sse])
  end
end

# function measure_cluster_sizes!(sim::Sim)
#   avg_cluster_sizes=zeros(macroint,length(sim.sse_configs))
#   Threads.@threads for coup in 1:length(sim.coupling_configs)
#     sse=sim.coup_to_sse_map[coup]
#     sim.sse_configs[sse].no_of_moves=128
#     diag_update!(sim.coupling_configs[coup],sim.sse_configs[sse])
#     construct_vertex_list!(sim.sse_configs[sse])
#     avg_cluster_sizes[sse]=Wolff_cluster_update!(sim.sse_configs[sse],1)
#     increase_op_leg_array!(sim.sse_configs[sse])
#   end
#   return avg_cluster_sizes
# end

# function assign_num_moves!(sim::Sim,avg_cluster_sizes::Array{macroint,1})
#   Threads.@threads for sse in 1:length(sim.sse_configs)
#     num_of_moves=Int64(round(2*sim.sse_configs[sse].num_of_op/avg_cluster_sizes[sse]))
#     sim.sse_configs[sse].no_of_moves=num_of_moves
#     println(sse," ",avg_cluster_sizes[sse], " ", num_of_moves)
#   end
# end

function thermalize!(sim::Sim)
  for i in 1:sim.sim_params[1].num_of_thermal
    # if i==sim.sim_params[1].num_of_thermal
    #   avg_cluster_sizes=measure_cluster_sizes!(sim)
    #   assign_num_moves!(sim,avg_cluster_sizes)
    #   return
    # end
    next_move!(sim)
    parallel_increase_op_leg_arrays!(sim)
  end
end

function parallel_measure!(sim)
  Threads.@threads for coup in 1:length(sim.coupling_configs)
  #for coup in 1:length(sim.coupling_configs)
    sse=sim.coup_to_sse_map[coup]
    measure_all!(sim.coupling_configs[coup],sim.sse_configs[sse],sim.obs[coup])
  end
end

function mc_run!(sim::Sim)
  for i=1:sim.sim_params[1].num_of_measure
    for j=1:sim.sim_params[1].num_of_sweep
      next_move!(sim)
    end
    parallel_measure!(sim)
  end
end
