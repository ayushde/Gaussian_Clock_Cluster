using Revise

include("Defs.jl")
include("AliasRand.jl")
include("GaussianCoeffs.jl")
include("MCConfig.jl")
include("MCMove.jl")
include("Measurements.jl")
include("SimParameters.jl")
include("Simulation.jl")


using Statistics, DependentBootstrap
using Plots
using LaTeXStrings
using LinearAlgebra
using BenchmarkTools
#using JETTest
using Bootstrap
using ThreadTools
using GJRand
using JLD2
using Dates
using Profile
using FFTW
using Tullio
using LoopVectorization
using FileIO
using JLD2

function bin_bootstrap_analysis(data;min_sample_size=4,func_boot=nothing,n_boot=1000)
    # get total length
    data_size=size(data)[1]
    # chop to closest power of 2
    chopped_data_size=2^floor(Int,log(2,data_size))
    chopped_data=data[end-chopped_data_size+1:end,:]
    # full data std
    if func_boot==nothing
        stds=[std(chopped_data)/sqrt(chopped_data_size)]
    else
        # bootstrap
        bs = bootstrap(func_boot,chopped_data, BasicSampling(n_boot))
        stds=[stderror(bs)[1] ]
    end
    bin_size=1
    while min_sample_size < div(chopped_data_size,bin_size)
        # bin size
        length_bin=div(chopped_data_size,bin_size)
        # binned data
        binned=reshape(chopped_data,(bin_size,length_bin,:))
        mean_binned= dropdims(mean(binned,dims=1),dims=1)
        # bin std
        if func_boot==nothing
            std_bin=std(mean_binned)/sqrt(length_bin)
        else
            # bootstrap
            bs = bootstrap(func_boot,mean_binned, BasicSampling(n_boot))
            std_bin = stderror(bs)[1]
        end
        # double bin size
        bin_size=bin_size*2
        push!(stds,std_bin)
    end
    stds
end

function init_res_Dict(NDIMS::Int64,g::Float64)
    res_dict = Dict(
                  "beta_arr"=>Float64[], "M_ET"=>Float64[],
                  "M2_ET"=>Float64[], "M2_ET_err"=>[], "M4_ET"=>Float64[], "M4_ET_err"=>Float64[],
                  "binder"=>Float64[], "binder_err"=>Float64[], "binder1"=>Float64[], "binder1_err"=>Float64[], "binder2"=>Float64[], "binder2_err"=>Float64[],
                  "chi_local"=>Float64[], "chi_local_err"=>Float64[],
                  "abs2_S_q_ET"=>Array{Float64, NDIMS}[], "abs2_S_q_ET_err"=>Array{Float64, NDIMS}[], "corr_length"=>Float64[], "corr_length_err"=>Float64[],
                  "energy"=>Float64[], "energy_err"=>Float64[], "g"=>Float64, "chi"=>Float64[], "chi_err"=>Float64[],
                  "cos_q_theta"=>Float64[],"cos_q_theta_err"=>Float64[], "G_r_ET"=>Array{Float64, NDIMS}[],"G_r_ET_err"=>Array{Float64, NDIMS}[]
                )
    res_dict["g"]=g
    return res_dict
end

function record!(res_dicts::Array{Dict{String,Any},1},sim::Sim{NDIMS, DIMS, QCLOCK, cis_q}) where {NDIMS, DIMS, QCLOCK,cis_q}
    Threads.@threads for coup in 1:length(sim.sse_configs)
    #for coup in 1:length(sim.sse_configs)
        coupling_config=sim.coupling_configs[coup]
        obs=sim.obs[coup]
        sim_params=sim.sim_params[coup]
        res_dict=res_dicts[coup]

        push!(res_dict["beta_arr"], coupling_config.beta)

        M2_ET = obs.M2_ET
        M4_ET = obs.M4_ET

        push!(res_dict["M2_ET"], mean(M2_ET))
        push!(res_dict["M2_ET_err"], bin_bootstrap_analysis(M2_ET))

        push!(res_dict["M4_ET"],  mean(M4_ET))
        push!(res_dict["M4_ET_err"], bin_bootstrap_analysis(M4_ET)[end])

        push!(res_dict["binder"], mean(M4_ET)/(mean(M2_ET)^2))
        push!(res_dict["binder_err"], bin_bootstrap_analysis([M2_ET M4_ET];func_boot=(x->mean(x[:,2])/mean(x[:,1])^2))[end]) #(input data, )

        num_of_half_measure = Int64(sim_params.num_of_measure/2)

        M2_ET = obs.M2_ET[1:num_of_half_measure]
        M4_ET = obs.M4_ET[1:num_of_half_measure]

        push!(res_dict["binder1"], mean(M4_ET)/(mean(M2_ET)^2))
        push!(res_dict["binder1_err"], bin_bootstrap_analysis([M2_ET M4_ET];func_boot=(x->mean(x[:,2])/mean(x[:,1])^2))[end])

        M2_ET = obs.M2_ET[num_of_half_measure + 1:end]
        M4_ET = obs.M4_ET[num_of_half_measure + 1:end]

        push!(res_dict["binder2"], mean(M4_ET)/(mean(M2_ET)^2))
        push!(res_dict["binder2_err"], bin_bootstrap_analysis([M2_ET M4_ET];func_boot=(x->mean(x[:,2])/mean(x[:,1])^2))[end])

        push!(res_dict["chi_local"], mean(obs.chi_local))
        push!(res_dict["chi_local_err"], bin_bootstrap_analysis(obs.chi_local)[end])

        abs2_S_q_ET = zeros(Float64, DIMS)
        abs2_S_q_ET_err = zeros(Float64, DIMS)
        G_r_ET = zeros(Float64, DIMS)
        G_r_ET_err = zeros(Float64, DIMS)
        for site in CartesianIndices(DIMS)
            abs2_S_q_ET[site] = mean(obs.abs2_S_q_ET[site])
            abs2_S_q_ET_err[site] = bin_bootstrap_analysis(obs.abs2_S_q_ET[site])[end]
            G_r_ET[site]=mean(abs.(obs.G_r[site]))
            G_r_ET_err[site]=bin_bootstrap_analysis(abs.(obs.G_r[site]))[end]
        end

        push!(res_dict["abs2_S_q_ET"], abs2_S_q_ET)
        push!(res_dict["abs2_S_q_ET_err"], abs2_S_q_ET_err)
        push!(res_dict["G_r_ET"],G_r_ET)
        push!(res_dict["G_r_ET_err"],G_r_ET_err)

        i=CartesianIndex(Tuple([Int64(j) for j in ones(NDIMS)]))
        g_0=obs.abs2_S_q_ET[i]/prod(DIMS)
        r=vcat([2],ones(NDIMS-1))
        i=CartesianIndex(Tuple([Int64(j) for j in r]))
        g_1=obs.abs2_S_q_ET[i]/prod(DIMS)
        push!(res_dict["corr_length"], (abs(mean(g_0)/mean(g_1) -1)^(0.5))*(1/(2*pi)))
        push!(res_dict["corr_length_err"],bin_bootstrap_analysis([g_0 g_1];func_boot=(x->(abs(mean(x[:,1])/mean(x[:,2]) -1)^(0.5))*(1/(2*pi))))[end])

        push!(res_dict["energy"],mean(obs.energy))
        push!(res_dict["energy_err"], bin_bootstrap_analysis(obs.energy)[end])

        push!(res_dict["chi"],mean(obs.chi_global))
        push!(res_dict["chi_err"], bin_bootstrap_analysis(obs.chi_global)[end])

        push!(res_dict["cos_q_theta"],mean(obs.cos_q_theta))
        push!(res_dict["cos_q_theta_err"], bin_bootstrap_analysis(obs.cos_q_theta)[end])
    end
end

function beta_double!(sim::Sim{NDIMS, DIMS, QCLOCK, cis_q}) where {NDIMS, DIMS, QCLOCK, cis_q}
    println(sim.coup_to_sse_map)
    Threads.@threads for coup in 1:length(sim.coupling_configs)
    #for coup in 1:length(sim.coupling_configs)
        sse=sim.coup_to_sse_map[coup]
        op_array=sim.sse_configs[sse].op_array
        for j=length(op_array):-1:1
                op=op_array[j]
                if op_array[j].op_type==field
                    op=Operator(op.op_type,op.site,op.dir,(op.colour[2],op.colour[1]))
                end
                push!(op_array, op)
            end
        num_of_op=2*sim.sse_configs[sse].num_of_op
        num_of_ising_op=2*sim.sse_configs[sse].num_of_ising_op
        beta=2*sim.coupling_configs[coup].beta
        q_config=copy(sim.sse_configs[sse].q_config)
        sim.coupling_configs[coup]=CouplingConfig(DIMS, QCLOCK, sim.sim_params[coup].J, sim.sim_params[coup].h, beta, sim.sim_params[coup].g)
        sim.sse_configs[sse]=SSEConfig(DIMS, QCLOCK, num_of_op, num_of_ising_op,op_array,sim.sim_params[coup].num_of_node_moves,sim.sim_params[coup].num_of_link_moves)
        sim.obs[coup]=ObsMagnetization(DIMS, cis_q, sim.sim_params[coup].num_of_measure, beta)
        sim.sse_configs[sse].q_config=q_config
    end
end

function run_PT_sim(sim_dicts::Array{Dict{String,Any},1})
    DIMS=(SimParams(sim_dicts[1]).L...,)
    max_log2_beta=SimParams(sim_dicts[1]).max_log2_beta
    NDIMS=length(DIMS)
    op_arrays = [[Operator(id_op, (ones(miniint, NDIMS)...,), zero(microint),(zero(microint),zero(microint)))] for sim_dict in sim_dicts]
    res_dicts=[init_res_Dict(NDIMS,sim_dict["g"]) for sim_dict in sim_dicts]
    sim_params=[SimParams(sim_dict) for sim_dict in sim_dicts]
    sim=Sim(sim_params,2.0,zeros(Int64,length(sim_dicts)),zeros(Int64,length(sim_dicts)),op_arrays)
    #sim=Sim(sim_params,1.0*2^max_log2_beta,zeros(Int64,length(sim_dicts)),zeros(Int64,length(sim_dicts)),op_arrays)
    for i in 1:max_log2_beta
        if i>1
            beta_double!(sim)
        end
        println(now()," Beginning Thermalization at beta=",1.0*2^i)
        #println(now()," Beginning Thermalization at beta=",1.0*2^max_log2_beta)
        thermalize!(sim)
        println(now()," Done Thermalization")
        mc_run!(sim)
        println(now()," Done MCMC at beta=",1.0*2^i)
        #println(now()," Done MCMC at beta=",1.0*2^max_log2_beta)
        record!(res_dicts,sim)
    end
    return res_dicts
end

function build_H_1d_clock(J, h, L, Q)
    w=cis(2*pi/Q)
    ws=[w^(q-1) for q=1:Q]
    Hmat=zeros(Complex{Float64},(Q^(L),Q^(L)))
    coeffs=gaussian_coeffs(Q)
    for state in 0:(Q^(L))-1
        current_state=zeros(L)
        c_state=state
        for i in 0:L-1
            current_state[L-i]=mod(c_state,Q)
            c_state=c_state-mod(c_state,Q)
            c_state=Int64(c_state/Q)
        end
        for i in 1:L
            nn=mod1(i+1,L)
            Hmat[state+1,state+1]+=J[1][i]*coeffs[Int64(mod(current_state[i]-current_state[nn],Q))+1]
        end
    end
    Hmat = real(Hmat)

    #transverse field term
    HTF_site=zeros(Float64,(Q,Q))
    for a in CartesianIndices((Q,Q))
        HTF_site[a]=coeffs[mod(a[1]-a[2],Q)+1]/sqrt(Q)
    end
    for site_ind=1:L
        h_c = h[site_ind]
        Hs = [1.0]
        for h_site=1:L
            Hs = h_site==site_ind ? h_c*kron(Hs, HTF_site) : kron(Hs,  Matrix{Float64}(I, Q, Q) )
        end
        Hmat += Hs
    end

    #order parameter
    Hmagsqr=zeros(Complex{Float64},(Q^L, Q^L))
    for site_ind=1:L
        Hw=[1.0+0.0im]
        for h_site=1:L
            Hw = h_site==site_ind ? kron(Hw, Matrix(Diagonal(ws)) ) : kron(Hw,  Matrix{Complex{Float64}}(I, Q, Q) )
        end

        Hmagsqr += Hw
    end
    Hmagsqr=Hmagsqr*adjoint(Hmagsqr) #Hmagsqr is diagonal

    -Hmat, real(Hmagsqr)
end


function ed_binder(e, v, beta, mag_mat)
    e0=e[1]
    em=e .- e0
    part=0.0
    m_av2=0.0
    m_av4=0.0
    mag_mat2=mag_mat*mag_mat
    for i=1:length(em)
        c_part = exp(-beta*em[i])
        part=part+c_part
        m_av2 += dot(v[:,i],mag_mat*v[:,i])*c_part
        m_av4 +=dot(v[:,i],mag_mat2*v[:,i])*c_part
    end
    m_av2=m_av2/part
    m_av4=m_av4/part
    return m_av4/(m_av2)^2
end

function ed_energy(e,beta)
    e0=e[1]
    es=e.-e0
    en=0.0
    part=0.0
    for (i,ec) in enumerate(es)
        c_part=exp(-beta*ec)
        part+=c_part
        en+=e[i]*c_part
    end
    en=en/part
    return en
end

function ed_chi(e,vs,beta,L,Q)
    w=cis(2*pi/Q)
    ws=[w^(q-1) for q=1:Q]
    Sz=zeros(Complex{Float64},(Q^L, Q^L))
    for site_ind=1:L
        Hw=[1.0+0.0im]
        for h_site=1:L
            Hw = h_site==site_ind ? kron(Hw, Matrix(Diagonal(ws)) ) : kron(Hw,  Matrix{Complex{Float64}}(I, Q, Q) )
        end

        Sz += Hw
    end
    part=0.0
    chi=0.0
    e0=e[1]
    es=e.-e0
    for (n,en) in enumerate(es)
        part+=exp(-beta*en)
        for (m,em) in enumerate(es)
            if abs(em-en)==0
                chi += beta*exp(-beta*en)*abs2(dot(vs[:,m],Sz*vs[:,n]))
            else
                chi += ((exp(-beta*en)-exp(-beta*em))*abs2(dot(vs[:,m],Sz*vs[:,n])))/(e[m]-e[n])
            end
        end
    end
    chi=chi/part
    return chi
end

gr()
Ls=[32]
Q=6
Js=range(0.8,length=8,stop=1.2)
#Js=[1.00]
fig_binders=plot(title="Binder")
fig_energies=plot(title="Energy")
fig_zi=plot(title="Correlation length")
fig_cos_q_theta=plot(title="Zq-U(1) Order Parameter")
fig_magsqr=plot(title="Magsqr")
#rng=GJRandRNG()
num_of_measure=2^3
num_of_thermal=2^8
num_of_sweep=2
num_of_node_moves=3
num_of_link_moves=3
num_of_PT_sweeps=1000
num_of_moves_between_PT=5
max_log2_betas=[10]

for (l,L) in enumerate(Ls)
    max_log2_beta=max_log2_betas[l]
    binders=Float64[]
    zis=Float64[]
    zis_std=Float64[]
    binders_std=Float64[]
    binder_eds=Float64[]
    energies=Float64[]
    energies_std=Float64[]
    energy_eds=Float64[]
    cos_q=Float64[]
    cos_q_std=Float64[]
    magsqrs=[]
    magsqrs_std=[]
    beta=2^max_log2_beta
    c_sim_datas=Dict{String,Any}[]
    for J in Js
        rng=GJRandRNG(UInt64(2*l))
        w0=1.0
        J_arr=Float64[]
        h_arr=Float64[]
        for j in 1:L
            h_j=1+(rand(rng)*2*w0)-w0
            J_j=J*(1+(rand(rng)*2*w0)-w0)
            push!(h_arr,h_j)
            push!(J_arr,J_j)
        end
        L_arr=Int64[L]
        J_arr=Array{Float64,1}[J_arr]
        h_arr=convert(Array{Float64,1}, h_arr)
        c_sim_data=Dict{String,Any}()
        c_sim_data["L"]=L_arr
        c_sim_data["QCLOCK"]=Q
        c_sim_data["J"]=J_arr
        c_sim_data["h"]=h_arr
        c_sim_data["max_log2_beta"]=max_log2_beta
        c_sim_data["num_of_thermal"]=num_of_thermal
        c_sim_data["num_of_sweep"]=num_of_sweep
        c_sim_data["num_of_measure"]=num_of_measure
        c_sim_data["num_of_node_moves"]=num_of_node_moves
        c_sim_data["num_of_link_moves"]=num_of_link_moves
        c_sim_data["num_of_PT_sweeps"]=num_of_PT_sweeps
        c_sim_data["num_of_moves_between_PT"]=num_of_moves_between_PT
        c_sim_data["g"]=J
        # println(h_arr)
        # println(J_arr)
        push!(c_sim_datas,c_sim_data)
        #res= @time run_PT_sim(c_sim_datas)
        #M2_errs=res[3]["M2_ET_err"]
        # (Hmat1,Magsqr)=build_H_1d_clock(J_arr,h_arr,L,Q)
        # (e,v)=eigen(Hmat1)L=convert(Array{Int64,1}, L)
             #h=
             #J=convert(Array{Array{Float64,1},1}, J)
        # binder_ed=ed_binder(e,v,beta,Magsqr)
        # push!(binder_eds,binder_ed)
        # energy_ed=ed_energy(e,beta)
        # push!(energy_eds,energy_ed)
        # println(c_sim_data)
        # bin_size=[2^i for i in 0:length(M2_errs[end])-1]
        # num_bins=[num_of_measure/2^i for i in 0:length(M2_errs[end])-1]
        # for b in 1:max_log2_beta
        #     varx=num_bins[1]*M2_errs[b][1]^2
        #     varxb=num_bins.*(M2_errs[b].^2)
        #     Rx=bin_size.*varxb/varx
        #     tau=0.5*(Rx.-1)
        #     println("taus = ",tau," for log2_beta = ",b)
        # end
        # varx=num_bins[1]*M2_errs[end][1]^2
        # varxb=num_bins.*(M2_errs[end].^2)
        # Rx=bin_size.*varxb/varx
        # println(0.5*(Rx.-1))
    end
    results=@time run_PT_sim(c_sim_datas)
    # M2_errs=results[3]["M2_ET_err"]
    # bin_size=[2^i for i in 0:length(M2_errs[end])-1]
    # num_bins=[num_of_measure/2^i for i in 0:length(M2_errs[end])-1]
    # for b in 1:max_log2_beta
    #     varx=num_bins[1]*M2_errs[b][1]^2
    #     varxb=num_bins.*(M2_errs[b].^2)
    #     Rx=bin_size.*varxb/varx
    #     tau=0.5*(Rx.-1)
    #     println("taus = ",tau," for log2_beta = ",b)
    # end
    # binders=[res["binder"][end] for res in results]
    # binders_std=[res["binder_err"][end] for res in results]
    # energies=[res["energy"][end] for res in results]
    # energies_std=[res["energy_err"][end] for res in results]
    # zis=[res["corr_length"][end] for res in results]
    # zis_std=[res["corr_length_err"][end] for res in results]
    # cos_q=[res["cos_q_theta"][end] for res in results]
    # cos_q_std=[res["cos_q_theta_err"][end] for res in results]
    # magsqrs=[res["M2_ET"][end]/L^(1.75) for res in results]
    # magsqrs_std=[res["M2_ET_err"][end]/L^(1.75) for res in results]

    # plot!(fig_binders,Js,binders,yerr=binders_std,xlabel=L"\h",ylabel=L"Binder_clock",label="mc L=$L Q=$Q",legend=:topright)
    # plot!(fig_binders,Js,binder_eds,xlabel=L"\h",ylabel=L"Binder_clock",label="ED L=$L Q=$Q",legend=:topright)
    # plot!(fig_energies,Js,energies,yerr=energies_std,xlabel=L"\h",ylabel=L"Energy_clock",label="mc L=$L Q=$Q",legend=:topright)
    # plot!(fig_energies,Js,energy_eds,xlabel=L"\h",ylabel=L"Energy_clock",label="ED L=$L Q=$Q",legend=:topright)
    # plot!(fig_zi,Js,zis,yerr=zis_std,xlabel=L"\h",ylabel=L"Correlation_length",label="mc L=$L Q=$Q",legend=:bottomright)
    # plot!(fig_cos_q_theta,Js,cos_q,yerr=cos_q_std,xlabel=L"\h",ylabel=L"Zq-U(1) Order Parameter",label="mc L=$L Q=$Q",legend=:topleft)
    # plot!(fig_magsqr,Js,magsqrs,yerr=magsqrs_std,xlabel=L"\h",ylabel=L"M^2",label="L=$L",legend=:bottomright)
end
#fig=plot(fig_binders,fig_energies,layout=grid(1,2),size = (1800,900))
