__precompile__()

module genAttrAGM
include("utils.jl")
include("genAGM.jl")
include("BigClam.jl")
using Graphs
using Random
using StatsBase
using Distributions
using  LinearAlgebra
using .utils
using .genAGM
using .BigClam

Base.@kwdef mutable struct AttrAgmParams
    radius::Float64 = 1
    att_centers::Union{Matrix{Float64},Nothing} = nothing
    weight_centers::Union{Matrix{Float64},Nothing} = nothing
    n_clusters::Int64 = 3
    # N::Union{Nothing, Int64} = nothing
    D_attr::Union{Nothing, Int64} = nothing
    D_weight::Union{Nothing, Int64} = nothing
    gen_edge_data_from_exp::Union{Nothing, Bool} = true ## If false, the edge data is sampled from ~N([F_u,F_v]*A,I)
     
    ### Choose one of the following to be true
    gen_net_given_probs::Union{Nothing, Bool} = true
    gen_net_given_density_scale::Union{Nothing, Bool} = false
    gen_net_given_density_n_edges::Union{Nothing, Bool} = false
    ###
    
    gen_attr_from_F::Union{Nothing, Bool} = false ## If false, uses the binary membership matrix M. If true, fits BigClam
    lambdas::Union{Nothing, Vector{Float64}} = nothing 

    """
    Params for AGM
    """
    agm_PNoCom::Union{Nothing, Float64} = -1.
    agm_DensityCoef::Union{Nothing, Float64} = 0.6  
    agm_ScaleCoef::Union{Nothing, Float64} = 1.3
    agm_TargetEdges::Union{Nothing, Int64} = 100
    agm_CProbV::Union{Nothing, Vector{Float64}} = nothing
    agm_use_MCMC::Union{Nothing, Bool} = true
    agm_n_iter_MCMC::Union{Nothing, Int64} = 1000

    """
    Params for BigClam
    """
    bigclam_n_iters::Union{Nothing, Int64} = 100
    bigclam_lr::Union{Nothing, Float64} = 0.005
    bigclam_display_ll::Union{Nothing, Bool} = false 
    bigclam_F_init::Union{Nothing, Matrix{Float64}} = nothing
    bigclam_use_linesearch::Union{Nothing, Bool} = false
    bigclam_threshold::Union{Nothing, Float64} = 0.01    
end


    function generate_edge_weights_exponential(G, 
                                            CProbV::Vector{Float64},
                                            lambdas::Vector{Float64},
                                            nodes_cmty::Vector{Set{Int64}},
                                            PNoCom::Float64
                                        )
        exp_dists = Vector{Exponential{Float64}}()
        K = length(lambdas)
        range_K = 1:K
        for lambda in lambdas
            push!(exp_dists,Exponential(1/lambda))
        end
        edges_data = Dict{Tuple{Int64,Int64},Vector{Float64}}()
        for edge in edges(G)
            u,v = edge.src,edge.dst
            if u > v
                u, v = v, u
            end
            u_cmtys = nodes_cmty[u]
            v_cmtys = nodes_cmty[v]
            intersection_uv = intersect(u_cmtys,v_cmtys)
            inter_uv_probs = zeros(Float64,K)
            if length(intersection_uv) > 0
                for k in intersection_uv
                    inter_uv_probs[k] = CProbV[k]
                end
            end
            if PNoCom > 0
                inter_uv_probs[K] = PNoCom
            end
            inter_uv_probs ./= sum(inter_uv_probs)
            cmty_sample = sample(range_K,Weights(inter_uv_probs))
            edges_data[(u,v)] = rand(exp_dists[cmty_sample],1)
            edges_data[(v,u)] = edges_data[(u,v)]
        end
        return edges_data
    end

    function generate_edge_data_from_GLM(G, M, A::Matrix{Float64},D::Int64)
        edges_data = Dict{Tuple{Int64,Int64},Vector{Float64}}()
        # I = identity(D)
        cov_matrix = Matrix{Float64}(I, D, D)  # Identity matrix of Float64 type
        for edge in edges(G)
            u,v = edge.src,edge.dst
            if u > v
                u, v = v, u
            end
            mean_ = (vcat(M[u,:],M[v,:])'*A)[1,:]
            norm_dist = MvNormal(mean_,cov_matrix)
            edges_data[(u,v)] = rand(norm_dist,1)[1,:]
            edges_data[(v,u)] = edges_data[(u,v)]
        end
        return edges_data
    end

    function generate_attr_data(M,benchmark_params::AttrAgmParams)
        centers = nothing
        if benchmark_params.att_centers != nothing
            centers = benchmark_params.att_centers
        else
            centers = utils.get_uniform_circle_coordinates(benchmark_params.n_clusters,
                                                           benchmark_params.radius)
            benchmark_params.att_centers = centers
        end
        means_ = M*centers
        N,D = size(means_)
        benchmark_params.D_attr = D
        cov_matrix = Matrix{Float64}(I, D, D)  # Identity matrix of Float64 type

        Y = zeros(size(means_))
        for i in 1:N
            norm_dist = MvNormal(means_[i,:],cov_matrix)
            Y[i,:] = rand(norm_dist,1)[:,1]
        end
        return Y
    end

    function genAttrAgm(CmtyVV::Vector{Vector{Int64}}, benchmark_params::AttrAgmParams)
        G = nothing

        if benchmark_params.gen_net_given_probs
            G = genAGM.genAgm(CmtyVV,
                            benchmark_params.agm_CProbV,
                            benchmark_params.agm_PNoCom,
                            benchmark_params.agm_use_MCMC,
                            benchmark_params.agm_n_iter_MCMC)
        
        elseif benchmark_params.gen_net_given_density_scale
            G = genAGM.genAgmDS(CmtyVV,
                                benchmark_params.agm_DensityCoef,
                                benchmark_params.agm_ScaleCoef)
        
        elseif benchmark_params.gen_net_given_density_n_edges
            G = genAGM.genAgmD(CmtyVV,
                                benchmark_params.agm_DensityCoef,
                                benchmark_params.agm_TargetEdges) 
        end
        N = nv(G)

        edges_data = Dict{Tuple{Int64,Int64},Vector{Float64}}()
        
        B = utils.get_binary_membership_matrix(CmtyVV,N)
        M = nothing
        if benchmark_params.gen_attr_from_F
            M = BigClam.fit(G, 
                            benchmark_params.n_clusters, 
                            benchmark_params.bigclam_n_iters,
                            benchmark_params.bigclam_lr,
                            benchmark_params.bigclam_display_ll,
                            benchmark_params.bigclam_F_init,
                            benchmark_params.bigclam_use_linesearch,
                            benchmark_params.bigclam_threshold
                            )
        else
            M = B
        end
        if benchmark_params.gen_edge_data_from_exp
            nodes_cmty = utils.get_nodes_cmty(CmtyVV,nv(G))
            edges_data = generate_edge_weights_exponential(G, 
                                                           benchmark_params.agm_CProbV, 
                                                           benchmark_params.lambdas, 
                                                           nodes_cmty, 
                                                           benchmark_params.agm_PNoCom)
        else
            D = size(benchmark_params.weight_centers)[2]
            edges_data = generate_edge_data_from_GLM(G, M, benchmark_params.weight_centers, D)
        end

        attr_data = generate_attr_data(M,benchmark_params)

        return G, edges_data, attr_data, B  
    end
end