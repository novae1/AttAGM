module MakeBenchmark

include("utils.jl")
include("genAGM.jl")
include("genAttrAGM.jl")
using .utils
using .genAGM
using .genAttrAGM
using NPZ
using Graphs
using BenchmarkTools
using Printf

"""
N -> number of nodes
K -> number of clusters
p_bipartite -> probability of a node belong to each cluster
PNoCom -> probability of two vertices without any common cluster connect to each other
D_attr -> Dimension of the attributes
D_weight -> Dimension of the weights
lambda_max -> If D_weight, sample the weights from a mixture of exponentials with parameters 1:l_max
"""

    function set_params(N,K,D_attr,D_weight,p_bipartite,P_in,PNoCom,lambda_max,attr_radius,weight_radius)::Tuple{Vector{Vector{Int64}}, genAttrAGM.AttrAgmParams}
        CmtyVV = utils.generate_bipartite_graph(N,K,nothing,p_bipartite,nothing) #G(n,m,p)
        
        params = genAttrAGM.AttrAgmParams()
        params.D_attr = D_attr
        params.D_weight = D_weight
        params.agm_PNoCom = PNoCom
        params.agm_use_MCMC = true
        params.agm_n_iter_MCMC = 1000
        params.agm_CProbV = repeat([P_in],K)
        n_cmty = PNoCom > 0 ? K+1 : K
        if D_weight > 2
            params.gen_edge_data_from_exp = false
            params.weight_centers = utils.generate_hypercube(2*K,D_weight) * weight_radius
        elseif D_weight == 2
            params.gen_edge_data_from_exp = false
            params.weight_centers = utils.get_uniform_circle_coordinates(2*K,weight_radius)
        else
            params.gen_edge_data_from_exp = true
            lambdas = collect(range(1,lambda_max,length=n_cmty))
            params.lambdas = lambdas
        end

        if params.D_attr > 2
            params.att_centers = utils.generate_hypercube(K,params.D_attr)
        else
            params.att_centers = utils.get_uniform_circle_coordinates(K,attr_radius)
        end
        params.n_clusters = K

        return CmtyVV, params
    end
    
    function generate(CmtyVV::Vector{Vector{Int64}},params::genAttrAGM.AttrAgmParams)
        # println("Starting AGM")
        # start = time()
        
        G,edges_data,attr_data,M = genAttrAGM.genAttrAgm(CmtyVV,params)
        
        # elapsed = time() - start
        # @printf("Finished AGM (%d nodes %d edges)\n",nv(G),ne(G))
        # (minutes, seconds) = fldmod(elapsed, 60)
        # (hours, minutes) = fldmod(minutes, 60)
        # @printf("Elapsed time: %02dh:%02dm:%0.2fs\n", hours, minutes, seconds)
        return G, edges_data, attr_data, M    
    end

    function generate(N,K,D_attr,D_weight,p_bipartite,P_in,PNoCom,lambda_max,attr_radius,weight_radius)

        CmtyVV, params = set_params(N,K,D_attr,D_weight,p_bipartite,P_in,PNoCom,lambda_max,attr_radius,weight_radius)
        println("Starting AGM")
        start = time()
        
        G,edges_data,attr_data,M = genAttrAGM.genAttrAgm(CmtyVV,params)
        
        elapsed = time() - start
        @printf("Finished AGM (%d nodes %d edges)\n",nv(G),ne(G))
        (minutes, seconds) = fldmod(elapsed, 60)
        (hours, minutes) = fldmod(minutes, 60)
        @printf("Elapsed time: %02dh:%02dm:%0.2fs\n", hours, minutes, seconds)
        return G, edges_data, attr_data, M
    end

end