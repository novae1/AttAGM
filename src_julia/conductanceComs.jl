__precompile__()

module conductanceComs

    using Distributions
    using Graphs

    mutable struct CommunityMembership
        F::Vector{Dict{Int, Float64}}  # Vector of dictionaries
        SumFV::Vector{Float64}      # Sum of values per community

        function CommunityMembership(num_nodes::Int, num_coms::Int)
            new([Dict{Int, Float64}() for _ in 1:num_nodes], Vector{Float64}())
        end
    end

    function AddCom(cm::CommunityMembership, NID::Int, CID::Int, Val::Float64)
        if haskey(cm.F[NID], CID)
            cm.SumFV[CID] -= cm.F[NID][CID]  # Remove old value from sum
        end
        cm.F[NID][CID] = Val  # Update community membership
        cm.SumFV[CID] = get(cm.SumFV, CID, 0.0) + Val  # Add new value to sum
    end

    function initialize_communities(G, cm::CommunityMembership, NumComs::Int)
        num_edges = ne(G)
        num_nodes = nv(G)
        cm.SumFV = zeros(Float64,NumComs)
        NIdPhiV = Vector{Tuple{Float64, Int}}()  # (Conductance, NodeID)
        InvalidNIDS = Set{Int}()
        ChosenNIDV = Int[]  # For debugging
        
        # Compute conductance for each node
        for u in 1:num_nodes
            neighbors_u = neighbors(G, u)  # Get neighbors
            phi = 0.0
            if length(neighbors_u) < 5
                phi = 1.0
            else
                NBCmty = Set(neighbors_u)
                push!(NBCmty, u)
                phi = get_conductance(G, NBCmty, num_edges)
            end
            push!(NIdPhiV, (phi, u))
        end
        
        # Sort nodes by conductance (ascending)
        sort!(NIdPhiV, by=x->x[1])
        println("Conductance computation completed.")

        # Select nodes with local minimum conductance
        cur_cid = 0
        for (phi, UID) in NIdPhiV
            if UID in InvalidNIDS
                continue
            end
            push!(ChosenNIDV, UID)  # For debugging
            
            # Add node to community
            AddCom(cm, UID, cur_cid, 1.0)
            for nbr in neighbors(G, UID)
                AddCom(cm, nbr, cur_cid, 1.0)
            end
            
            # Mark neighbors as invalid for next considerations
            for nbr in neighbors(G, UID)
                push!(InvalidNIDS, nbr)
            end
            
            cur_cid += 1
            if cur_cid >= NumComs
                break
            end
        end
        
        if NumComs > cur_cid
            println("$(NumComs - cur_cid) communities needed to fill randomly")
        end

        # Assign a member to zero-member communities
        for c in keys(cm.SumFV)
            if cm.SumFV[c] == 0.0
                com_sz = 10
                for _ in 1:com_sz
                    uid = rand(1:num_nodes)
                    AddCom(cm, uid, c, rand())
                end
            end
        end
    end

    function get_conductance(G, CmtyS::Set{Int}, Edges::Int)
        Edges2 = Edges >= 0 ? 2 * Edges : ne(G)
        Vol, Cut = 0, 0
        Phi = 0.0

        for node in CmtyS
            if !(has_vertex(G,node))  # Check if node exists in the graph
                continue
            end
            for neighbor in neighbors(G, node)
                if !(neighbor in CmtyS)
                    Cut += 1
                end
            end
            Vol += length(neighbors(G, node))
        end

        # Compute conductance
        if Vol != Edges2
            if 2 * Vol > Edges2
                Phi = Cut / (Edges2 - Vol)
            elseif Vol == 0
                Phi = 0.0
            else
                Phi = Cut / Vol
            end
        elseif Vol == Edges2
            Phi = 1.0
        end
        return Phi
    end
end
