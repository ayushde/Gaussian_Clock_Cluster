using Base.Cartesian

@generated function hop(site::CartesianIndex{NDIMS}, dir::microint, dims::NTuple{NDIMS, Int64}) where NDIMS
  quote
    @ncall $NDIMS CartesianIndex i->
      if i == dir
        site[i] == dims[i] ? 1 : site[i] + 1
      else
        site[i]
      end
  end
  # Base.setindex(site, dir, site[dir] == dims[dir] ? 1 : site[dir] + 1)
end

function check_config(new_config::Array{Int8,NDIMS},op_array::Array{Operator{NDIMS},1},sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  old_config=copy(new_config)
  for op in op_array
    op_site=CartesianIndex(op.site)
    if op.op_type==field && op.colour[1]!=op.colour[2]
      old_config[op_site]=op.colour[2]
    elseif op.op_type==Ising
      nn_site=hop(op_site,op.dir,DIMS)
      old_config[op_site]=op.colour[1]
      old_config[nn_site]=op.colour[2]
    end
  end
  if old_config!=new_config
    println("updated old config = ", old_config)
    println("New config = ", new_config)
    sleep(1)
  end
end

function check_first_last(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS,DIMS,QCLOCK} #Pass
  for site in CartesianIndices(DIMS)
    last_leg=sse_config.last_spin_leg[site]
    l_p=div(last_leg,5)
    leg_index=mod(last_leg,5)
    if l_p==0
      continue
    end
    l_op=sse_config.op_array[l_p]
    l_node=mod1(leg_index,2)
    first_leg=sse_config.first_spin_leg[site]
    f_p=div(first_leg,5)
    f_node=mod(first_leg,5)
    f_op=sse_config.op_array[f_p]
    if f_op.colour[f_node]!=l_op.colour[l_node] #&& (f_op.op_type==field && l_op.op_type==field)
      println("First operator ",f_op.colour," ",f_op.op_type," ", f_node, " ",f_op.site, " ",f_p)
      println("Last operator ",l_op.colour," ",l_op.op_type," ", l_node, " ",l_op.site, " ", l_p)
      println(" ")
      sleep(1)
    end
  end
end

function check_vertex_boundary(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS,DIMS,QCLOCK} #Pass
  for site in CartesianIndices(DIMS)
    last_leg=sse_config.last_spin_leg[site]
    if last_leg==0
      continue
    end
    l_p=div(last_leg,5)
    leg_index=mod(last_leg,5)
    if sse_config.first_spin_leg[site]!=sse_config.leg_array[l_p,leg_index]
      println("We have a problem")
      sleep(1)
    end
  end
end

function operator_stats(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS,DIMS,QCLOCK}
  tot_ops=0
  field_ops=0
  bond_ops=0
  field_array=zeros(QCLOCK)
  bond_array=zeros(QCLOCK)
  for op in sse_config.op_array
    tot_ops+=1
    delc=mod(op.colour[2]-op.colour[1],QCLOCK)
    if op.op_type==field
      field_ops+=1
      field_array[delc+1]+=1
    elseif op.op_type==Ising
      bond_ops+=1
      bond_array[delc+1]+=1
    end
  end
  println("tot_ops=",tot_ops," Ising_ops=",bond_ops, " field_ops=",field_ops)
  println("Ising_fraction=",bond_ops/tot_ops," field_fraction=",field_ops/tot_ops)
  println("field_array= ",field_array)
  println("bond_array= ",bond_array)
  println("field_array_frac= ",field_array./tot_ops)
  println("bond_array_frac= ",bond_array./tot_ops)
end

function print_op(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS,DIMS,QCLOCK}
  for op in sse_config.op_array
    println(op.op_type, " ", op.colour[1], " ", op.colour[2], " ", op.site)
    sleep(1)
  end
end

function diag_update!(coupling_config::CouplingConfig{NDIMS, DIMS, QCLOCK}, sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  for move in 1:3
    for (p, op) in enumerate(sse_config.op_array)
      if op.op_type == id_op # if the operator is the identity operator
        # p_insert = probability of replacing identity operator with non-identity diagonal operator
        p_insert = coupling_config.beta*coupling_config.tot_coupling_const
        if rand(sse_config.rng)*(length(sse_config.op_array) - sse_config.num_of_op) < p_insert
          #index = searchsortedfirst(sse_config.p_op_cum, rand(sse_config.rng))
          index=alias_rand(coupling_config.U,coupling_config.Q,sse_config.rng)
          if index <= length(coupling_config.h) # field-diag operator case
            if rand(sse_config.rng)<sse_config.weights[1]
              op_type = field # always insert field-diag operator

              op_site = CartesianIndices(DIMS)[index]

              op_colour = sse_config.q_config[op_site]

              op_site_t = ntuple(i->miniint(op_site[i]), NDIMS)

              sse_config.op_array[p] = Operator(op_type, op_site_t, zero(microint), (op_colour,op_colour))

              sse_config.num_of_op += 1
            end
          else # Ising operator case
            index = index - length(coupling_config.h) # find the correct index for Ising operator

            op_sitedir = CartesianIndices((DIMS..., NDIMS))[index]

            op_site_t = ntuple(i->miniint(op_sitedir[i]), NDIMS)
            op_dir = op_sitedir[NDIMS + 1]

            nn_site = hop(CartesianIndex(op_site_t), microint(op_dir), DIMS)

            if rand(sse_config.rng)<sse_config.weights[mod(sse_config.q_config[nn_site]-sse_config.q_config[CartesianIndex(op_site_t)],QCLOCK)+1]
              op_colour1 = sse_config.q_config[CartesianIndex(op_site_t)]
              op_colour2 = sse_config.q_config[nn_site]
              op_type = Ising # insert the chosen Ising operator
              sse_config.op_array[p] = Operator(op_type, op_site_t, microint(op_dir),(op_colour1,op_colour2))
              sse_config.num_of_op += 1
              sse_config.num_of_ising_op +=1
            end
          end
        # sse_config.leg_array is yet to be updated
        end

      elseif op.op_type == Ising || (op.op_type == field && op.colour[1]==op.colour[2])# if the operator is a diagonal operator
        p_remove = length(sse_config.op_array) - sse_config.num_of_op + 1
        if rand(sse_config.rng)*(coupling_config.beta*coupling_config.tot_coupling_const)< p_remove
          op_type = id_op
          sse_config.op_array[p] = Operator(op_type, op.site, zero(microint),(zero(microint),zero(microint)))
          sse_config.num_of_op -= 1
          if op.op_type == Ising
            sse_config.num_of_ising_op -= 1
          end
        end
      else  # :(op.op_type = (field, n)) with n != 0
        sse_config.q_config[CartesianIndex(op.site)] = op.colour[2]
      end
    end
  end
end

function increase_op_leg_array!(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  dM = convert(Int64, 1+round(sse_config.num_of_op*5/4 - length(sse_config.op_array)) )

  for i=1:dM # increase op_array if necessary
    push!(sse_config.op_array, Operator(id_op, (ones(miniint, NDIMS)...,), zero(microint),(zero(microint),zero(microint))) ) # Operator((id_op, 0), (1,...,1), 0)
  end
  if dM > 0
    new_op_size = length(sse_config.op_array)
    sse_config.leg_array = fill(zero(macroint), (new_op_size, 4))
    sse_config.node_visited_array=fill(zero(microint),(new_op_size, 2))
  end
end

function construct_vertex_list!(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  fill!(sse_config.first_spin_leg, zero(macroint))
  fill!(sse_config.last_spin_leg, zero(macroint))

  for (p, op) in enumerate(sse_config.op_array)
    if op.op_type == field
      last_leg = sse_config.last_spin_leg[CartesianIndex(op.site)] # the last leg at op.site
      if last_leg == 0 # if last_leg has never been updated, then first_leg at the same site has never been updated either
        # update (for the first time & also the last time) first_spin_leg at op.site
        sse_config.first_spin_leg[CartesianIndex(op.site)] = 5p+1

      else # if we already have a valid last_leg, this gives op's leg #1
        sse_config.leg_array[p, 1] = last_leg
        sse_config.leg_array[div(last_leg,5),mod(last_leg,5)] = 5p+1 # also updates upward leg
      end

      sse_config.last_spin_leg[CartesianIndex(op.site)] = 5p+2 # always updates last_spin_leg
    elseif op.op_type == Ising
      # leg #1
      last_leg = sse_config.last_spin_leg[CartesianIndex(op.site)] # the last leg at op.site
      if last_leg == 0 # if last_leg has never been updated, then first_leg at the same site has never been updated either
        # update (for the first time & also the last time) the first_spin_leg at op.site
        sse_config.first_spin_leg[CartesianIndex(op.site)] = 5p+1

      else # if we already have a valid last_leg, this gives the op's leg #1
        sse_config.leg_array[p,1] = last_leg
        sse_config.leg_array[div(last_leg,5),mod(last_leg,5)] = 5p+1 # also updates upward leg
      end

      sse_config.last_spin_leg[CartesianIndex(op.site)] = 5p+3 # always updates last_spin_leg
      # leg #1 updated
      # leg #2
      nn_site = hop(CartesianIndex(op.site), op.dir, DIMS)
      last_leg = sse_config.last_spin_leg[nn_site] # last_leg at nn_site
      if last_leg == 0 # if last_leg has never been updated, then first_leg at the same site has never been updated either
        # update (for the first time & also the last time) first_spin_leg at nn_site
        sse_config.first_spin_leg[nn_site] = 5p+2

      else # if we already have a valid last_leg, this gives op's leg #2
        sse_config.leg_array[p, 2] = last_leg
        sse_config.leg_array[div(last_leg,5),mod(last_leg,5)] = 5p+2 # also updates upward leg
      end

      sse_config.last_spin_leg[nn_site] = 5p+4 # always updates last_spin_legfunction construct_vertex_list!(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  fill!(sse_config.first_spin_leg, zero(macroint))
  fill!(sse_config.last_spin_leg, zero(macroint))

  for (p, op) in enumerate(sse_config.op_array)
    if op.op_type == field
      last_leg = sse_config.last_spin_leg[CartesianIndex(op.site)] # the last leg at op.site
      if last_leg == 0 # if last_leg has never been updated, then first_leg at the same site has never been updated either
        # update (for the first time & also the last time) first_spin_leg at op.site
        sse_config.first_spin_leg[CartesianIndex(op.site)] = 5p+1

      else # if we already have a valid last_leg, this gives op's leg #1
        sse_config.leg_array[p, 1] = last_leg
        sse_config.leg_array[div(last_leg,5),mod(last_leg,5)] = 5p+1 # also updates upward leg
      end

      sse_config.last_spin_leg[CartesianIndex(op.site)] = 5p+2 # always updates last_spin_leg
    elseif op.op_type == Ising
      # leg #1
      last_leg = sse_config.last_spin_leg[CartesianIndex(op.site)] # the last leg at op.site
      if last_leg == 0 # if last_leg has never been updated, then first_leg at the same site has never been updated either
        # update (for the first time & also the last time) the first_spin_leg at op.site
        sse_config.first_spin_leg[CartesianIndex(op.site)] = 5p+1

      else # if we already have a valid last_leg, this gives the op's leg #1
        sse_config.leg_array[p,1] = last_leg
        sse_config.leg_array[div(last_leg,5),mod(last_leg,5)] = 5p+1 # also updates upward leg
      end

      sse_config.last_spin_leg[CartesianIndex(op.site)] = 5p+3 # always updates last_spin_leg
      # leg #1 updated
      # leg #2
      nn_site = hop(CartesianIndex(op.site), op.dir, DIMS)
      last_leg = sse_config.last_spin_leg[nn_site] # last_leg at nn_site
      if last_leg == 0 # if last_leg has never been updated, then first_leg at the same site has never been updated either
        # update (for the first time & also the last time) first_spin_leg at nn_site
        sse_config.first_spin_leg[nn_site] = 5p+2

      else # if we already have a valid last_leg, this gives op's leg #2
        sse_config.leg_array[p, 2] = last_leg
        sse_config.leg_array[div(last_leg,5),mod(last_leg,5)] = 5p+2 # also updates upward leg
      end

      sse_config.last_spin_leg[nn_site] = 5p+4 # always updates last_spin_leg
      # leg #2 updated
    end
  end # updated 1) first_spin_leg, 2) last_spin_leg, and 3) leg_array except ones on the boundary
  # boundary condition
  for site in CartesianIndices(DIMS)
    last_leg = sse_config.last_spin_leg[site]
    if last_leg != 0 # if last_leg is not null
      first_leg = sse_config.first_spin_leg[site] # first_leg is not null either!

      sse_config.leg_array[div(last_leg,5),mod(last_leg,5)] = first_leg
      sse_config.leg_array[div(first_leg,5),mod(first_leg,5)] = last_leg
    end
  end
end

function flip_colour(colour::microint,flip_axis::Int64,QCLOCK::Int64)

      # leg #2 updated
    end
  end # updated 1) first_spin_leg, 2) last_spin_leg, and 3) leg_array except ones on the boundary
  # boundary condition
  for site in CartesianIndices(DIMS)
    last_leg = sse_config.last_spin_leg[site]
    if last_leg != 0 # if last_leg is not null
      first_leg = sse_config.first_spin_leg[site] # first_leg is not null either!

      sse_config.leg_array[div(last_leg,5),mod(last_leg,5)] = first_leg
      sse_config.leg_array[div(first_leg,5),mod(first_leg,5)] = last_leg
    end
  end
end

function flip_colour(colour::microint,flip_axis::Int64,QCLOCK::Int64)
  if mod(QCLOCK,2)==1
    return mod1(2*flip_axis-colour,QCLOCK)
  else
    if mod(flip_axis,2)==0
      return mod1(flip_axis-colour,QCLOCK)
    else
      alpha=Int64((flip_axis+1)/2)
      return mod1(2*alpha-1-colour,QCLOCK)
    end
  end
end

function single_spin_update!(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  for i in 1:length(sse_config.op_array)
    p1=rand(sse_config.rng,1:length(sse_config.op_array))
    op1=sse_config.op_array[p1]
    if op1.op_type==field
      node1 = rand(sse_config.rng,1:2)
      p2=div(sse_config.leg_array[p1,node1],5)
      if p2 == p1 #This demands a diagonal field operator
        colour=op1.colour[1]
        flip_axis=rand(sse_config.rng,1:QCLOCK)
        f_colour=flip_colour(colour,flip_axis,QCLOCK)
        sse_config.op_array[p1]=Operator(op1.op_type,op1.site,op1.dir,(microint(f_colour),microint(f_colour)))
        continue
      end
      op2=sse_config.op_array[p2]
      if op2.op_type==field
        node2 = node1 == 1 ? 2 : 1 # For op1 node1 is the active node and node2 is passive and vice versa for op2
        flip_axis=rand(sse_config.rng,1:QCLOCK)
        colour=op1.colour[node1]
        f_colour=flip_colour(colour,flip_axis,QCLOCK)
        p_flip=(sse_config.weights[mod(op1.colour[node2]-f_colour,QCLOCK)+1]*sse_config.weights[mod(op2.colour[node1]-f_colour,QCLOCK)+1])
        if rand(sse_config.rng)*(sse_config.weights[mod(op1.colour[node2]-colour,QCLOCK)+1]*sse_config.weights[mod(op2.colour[node1]-colour,QCLOCK)+1])<p_flip
          new_colour1 = node1 == 1 ? (microint(f_colour),op1.colour[node2]) : (op1.colour[node2],microint(f_colour))
          new_colour2 = node2 == 1 ? (microint(f_colour),op2.colour[node1]) : (op2.colour[node1],microint(f_colour))
          sse_config.op_array[p1]=Operator(op1.op_type,op1.site,op1.dir,new_colour1)
          sse_config.op_array[p2]=Operator(op2.op_type,op2.site,op2.dir,new_colour2)
        end
      end
    end
  end
end

function flip_node!(p::Int64,node::Int64,sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  if sse_config.cluster_flip_axis==0
    return
  end
  op=sse_config.op_array[p]
  if mod(QCLOCK,2)==1 #Case1 Odd Q
    f_op_colour= node==1 ? (microint(mod1(2*sse_config.cluster_flip_axis-op.colour[1],QCLOCK)),op.colour[2]) : (op.colour[1],microint(mod1(2*sse_config.cluster_flip_axis-op.colour[2],QCLOCK)))
  else
    if mod(sse_config.cluster_flip_axis,2)==0 #Case2B Primed Axis
       f_op_colour= node==1 ? (microint(mod1(sse_config.cluster_flip_axis-op.colour[1],QCLOCK)),op.colour[2]) : (op.colour[1],microint(mod1(sse_config.cluster_flip_axis-op.colour[2],QCLOCK)))
    else #Case2A Unprimed
      alpha=Int64((sse_config.cluster_flip_axis+1)/2)
      f_op_colour= node==1 ? (microint(mod1(2*alpha-1-op.colour[1],QCLOCK)),op.colour[2]) : (op.colour[1],microint(mod1(2*alpha-1-op.colour[2],QCLOCK)))
    end
  end
  sse_config.op_array[p]=Operator(op.op_type,op.site,op.dir,f_op_colour)
end

function flip_site!(site::CartesianIndex{NDIMS},sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  if sse_config.cluster_flip_axis==0
    return
  end
  if mod(QCLOCK,2)==1 #Case1 Odd Q
    sse_config.q_config[site]=microint(mod1(2*sse_config.cluster_flip_axis-sse_config.q_config[site],QCLOCK))
  else
    if mod(sse_config.cluster_flip_axis,2)==0 #Case2B Primed Axis
      sse_config.q_config[site]=microint(mod1(2*sse_config.cluster_flip_axis-sse_config.q_config[site],QCLOCK))
    else #Case2A Unprimed
      alpha=Int64((sse_config.cluster_flip_axis+1)/2)
      sse_config.q_config[site]=microint(mod1(2*alpha-1-sse_config.q_config[site],QCLOCK))
    end
  end
end

function add_neighbour_node!(p::Int64,node::Int64,sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  op=sse_config.op_array[p]
  next_node= node==1 ? 2 : 1
  if sse_config.node_visited_array[p,next_node]==1 #Checking if next_node has been visited
    return
  end
  if op.op_type==field #Field Operator Case
    if rand(sse_config.rng) <= 1-0.5*(sse_config.weights[mod(op.colour[2]-op.colour[1],QCLOCK)+1])/(sse_config.weights[1])
      push!(sse_config.cluster_leg_array,5p+next_node)
    end
  elseif rand(sse_config.rng) <= 1-0.5*(sse_config.weights[mod(op.colour[2]-op.colour[1],QCLOCK)+1])/(sse_config.weights[1]) #Ising Operator Case, adding neighbour with appropriate probability
    push!(sse_config.cluster_leg_array,5p+next_node)
  end
end
function update_sites!(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  for site in CartesianIndices(DIMS)
    if sse_config.first_spin_leg[site]==0
      continue
    end
    p=div(sse_config.first_spin_leg[site],5)
    leg=mod(sse_config.first_spin_leg[site],5)
    op=sse_config.op_array[p]
    sse_config.q_config[site]=op.colour[leg]
  end
end

function percolate_cluster!(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  while !isempty(sse_config.cluster_leg_array)
    leg = pop!(sse_config.cluster_leg_array)
    p=div(leg,5)
    leg_index=mod(leg,5)
    node=mod1(leg_index,2)
    if sse_config.node_visited_array[p,node]==1 #Checking if node has been visited
      continue
    end
    flip_node!(p,node,sse_config) #Flipping current node
    op=sse_config.op_array[p]
    add_neighbour_node!(p,node,sse_config) #propose to add neighbor node to the cluster
    push!(sse_config.cluster_leg_array,sse_config.leg_array[p,leg_index]) #Add to cluster non-neighbour connected node
    if op.op_type==Ising
      other_leg=mod1(leg_index+2,4)
      push!(sse_config.cluster_leg_array,sse_config.leg_array[p,other_leg]) #Add to cluster second non-neighbour connected node
    end
    sse_config.node_visited_array[p,node]=1 #Marking node as visited
  end
end

function find_link_seed(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  p_init=rand(sse_config.rng,1:length(sse_config.op_array))
  site=rand(sse_config.rng,CartesianIndices(DIMS))
  first_leg=sse_config.first_spin_leg[site]
  if first_leg==0
    return 0,site
  end
  p_current=div(first_leg,5)
  if p_init < p_current
    last_leg=sse_config.last_spin_leg[site]
    return div(last_leg,5),mod(last_leg,5)
  end
  op=sse_config.op_array[p_current]
  if op.op_type==field
    current_leg=2 #Field Operator
  else #Ising Case
    current_leg = op.site == site ? 3 : 4
  end
  p_next=p_current
  next_leg=current_leg
  while p_next < p_init
    #println("yolo")
    p_current=p_next
    current_leg=next_leg
    next_index=sse_config.leg_array[p_current,current_leg]
    #println(p_current," ",current_leg," ",next_index)
    p_next=div(next_index,5)
    if p_next<=p_current
      break
    end
    op=sse_config.op_array[p_next]
    if op.op_type==field
      next_leg=2 #Field Operator
    else #Ising Case
      next_leg = op.site == site ? 3 : 4
    end
  end
  #println("p_current= ",p_current," current_leg= ",current_leg)
  return p_current,current_leg
end

function node_cluster_update!(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  fill!(sse_config.node_visited_array,zero(microint))
  p=rand(sse_config.rng,(1:length(sse_config.op_array)))
  op=sse_config.op_array[p]
  if op.op_type==id_op
    return
  end
  sse_config.cluster_flip_axis=rand(sse_config.rng,1:QCLOCK)
  node=rand(sse_config.rng,(1:2))
  push!(sse_config.cluster_leg_array,macroint(5p+node))
  percolate_cluster!(sse_config)
end

function link_cluster_update!(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  fill!(sse_config.node_visited_array,zero(microint))
  a=find_link_seed(sse_config)
  #println("a=",a)
  (p,leg)=a
  sse_config.cluster_flip_axis=rand(sse_config.rng,1:QCLOCK)
  if p==0
    site=leg
    flip_site!(site,sse_config)
    return
  end
  push!(sse_config.cluster_leg_array,macroint(5p+leg))
  percolate_cluster!(sse_config)
end

function Wolff_cluster_update!(sse_config::SSEConfig{NDIMS, DIMS, QCLOCK}) where {NDIMS, DIMS, QCLOCK}
  for i in 1:sse_config.no_of_node_moves
    node_cluster_update!(sse_config)
  end
  for i in 1:sse_config.no_of_link_moves
    link_cluster_update!(sse_config)
  end
end
