import numpy as np
from scipy.optimize import fsolve
import networkx as nx
from networkx import bipartite
import subprocess
from scipy.stats import powerlaw, expon
import bigclam as bc
from sklearn.preprocessing import normalize
from copy import deepcopy

class attrAGM():
    def __init__(self,membership_matrix=None,\
                    radius=1,\
                    att_centers = None,\
                    weight_centers = None,\
                    num_communities = None\
                    ):
                    
        self.F = membership_matrix
        self.radius = radius
        ## att_centers must have shape K x D, where K is the number
        #  of communities and D the number of dimensions.
        # If not specified, then the centers are taken from unit circle
        self.att_centers=att_centers
        ## weight_centers must have shape K x D, where K is the number
        #  of communities
        self.weight_centers=weight_centers
        self.n_clusters = num_communities
        self.bigclam_use_linesearch = False
        self.gen_attr_from_F = False ## If false, uses the binary membership matrix M. If true, fits BigClam
    ### Generate the network according to the NULL model
    def gen_bipartite_net(self,num_nodes,n_clusters,num_edges=None,bipartite_prob=None):
        B = None
        if bipartite_prob is not None:
            B = bipartite.random_graph(num_nodes, n_clusters, bipartite_prob)
        elif num_edges is not None:
            B = bipartite.gnmk_random_graph(num_nodes, n_clusters, num_edges)
        #top nodes are the nodes that compose the network
        top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}  
        bottom_nodes = set(B) - top_nodes #bottom_nodes are the community nodes
        membership_matrix = np.zeros((len(top_nodes),n_clusters))
        with open("bipartite.txt",'w') as f:
            i = 0
            for node in bottom_nodes:
                ## Write the file for agmgen input
                ## each line contains the nodes tha belongs to cmty "i"
                group = [str(n) for n in B.neighbors(node)]
                str_group = ' '.join(group) + "\n"
                f.write(str_group)
                
                ## store the information in membership matrix
                for n in B.neighbors(node):
                    membership_matrix[n,i] = 1
                i+=1
        with open('membership_matrix.npy', 'wb') as f:
            np.save(f, membership_matrix)
        return B
            
    def generate_network(self, bipartite_file, output_file,\
                                a = 0.6, c = 1.3,\
                                lambdas = None,\
                                probs=None,\
                                pn=-1):
        """
        Parameters:
            -i: Community affiliation data (one group per line). 
                Use 'DEMO' to consider a simple case where Nodes 0-24 belong to 
                first community, and nodes 15-39 belong to the second community.
                Also check 'community_affilications.txt' for an example.
            -o: Output filename prefix (The edges of the generated graph).
            -a: Power-law coefficient of the distribution of edge probability 
                inside each community.
            -c: Scaling constant for the edge probability.
            -l: The lambdas for exponential distributed weights
            -probs: the list with probability of edge inside each community
            -pn: probability of non-edge
        """
        
        if probs is not None:
            probs_ = [str(i) for i in probs]
            str_probs = ",".join(probs_)
        else:
            str_probs= ""
            
        if lambdas is not None:
            lambdas_ = [str(i) for i in lambdas]
            str_lambdas = ",".join(lambdas_)
        else:
            str_lambdas= ""
            
        subprocess.run(["./agmgen",\
                            f"-i:{bipartite_file}",\
                            f"-a:{a}",\
                            f"-c:{c}",\
                            f"-o:{output_file}",\
                            "-l:%s"%str_lambdas,\
                            "-p:%s"%str_probs,\
                            f"-pn:{pn}"]
                       )
        G = nx.read_edgelist(output_file,\
            create_using=nx.Graph,\
            nodetype = int,\
            edgetype=int,\
            data=(("weight", float),)
            )
        M = self.get_binary_membership_matrix(bipartite_file)
        #M = np.load('membership_matrix.npy')
        for u in G.nodes:
            G.nodes[u]["membership"] = M[u,:]
        GG = nx.relabel.convert_node_labels_to_integers(G, 
                                                        first_label=0, 
                                                        ordering='default',
                                                        label_attribute="old_id") 
        edge_data_exists = True if str_lambdas != "" else False
        M_sub = np.zeros((GG.number_of_nodes(),M.shape[1]))
        for u in GG.nodes():
            M_sub[u,:] = M[GG.nodes[u]["old_id"], : ]
        return GG, edge_data_exists, M_sub
    
    def get_membership_matrix_from_net(self,G,n_clusters,n_steps):
        F = bc.train_efficient(G,
                               n_clusters,
                               iterations = n_steps,
                               use_line_search=self.bigclam_use_linesearch)
        return F
    
    def get_binary_membership_matrix(self,bipartite_file="bipartite.txt"):
        max_ = 0
        with open(bipartite_file, "r") as f:
            candidates = []
            for line in f:
                candidates.append(max([int(x) for x in line.split()]))
        max_ = max(candidates)
        k = len(candidates)
        M = np.zeros((max_+1,k))
        with open(bipartite_file, "r") as f:
            i=0
            for line in f:
                for u in line.split():
                    M[int(u),i] = 1
                i+=1 #iterate the community number
        return M
    
    # Specify equidistant centers of covariates around a circle of radius R
    def get_unit_circle_coordinates(self):
        centers = []
        for i in range(self.n_clusters):
            centers.append([self.radius*np.cos(2*np.pi*i/self.n_clusters),\
                          self.radius*np.sin(2*np.pi*i/self.n_clusters)])
        return np.array(centers)
    
    def generate_attributes(self,F):
        centers = None
        if self.att_centers is not None:
            centers = self.att_centers
        else: 
            centers = self.get_unit_circle_coordinates()
            self.att_centers = centers
        #means = normalize(F,norm="l1")@centers
        means = F@centers
        D = means.shape[1]
        Y = np.zeros(means.shape)
        for i in range(Y.shape[0]):
            Y[i,:] = np.random.multivariate_normal(means[i,:],np.eye(D))
        return Y
    
    ## Generate Benchmark where the edges follows a joint probability distribution
    #  given by bernoulli and weights from exponential familly 
    def generate_benchmark(self,bipartite_file="bipartite.txt",\
                            output_file="agm.txt",\
                            a=0.6,\
                            c=1.3,\
                            lambdas=None,\
                            probs=None,\
                            pn=-1,\
                            n_steps=100):
        G, edge_data_exists, M = self.generate_network(bipartite_file, output_file,\
                                a = a, c = c,\
                                lambdas = lambdas,\
                                probs= probs,\
                                pn=pn)
        if self.n_clusters is None:
            with open(bipartite_file) as file:
                 lines = file.readlines()
                 self.n_clusters = len(lines)
        F = Y = None
        if self.gen_attr_from_F:
            F = self.get_membership_matrix_from_net(G,self.n_clusters,n_steps)
        else:
            F = M
        Y = self.generate_attributes(F)
        E = None
        if not edge_data_exists and self.weight_centers is not None:
            D = self.weight_centers.shape[1]
            E = np.zeros((len(G.edges),self.weight_centers.shape[1]))
            i = 0
            for u,v in G.edges:
                min_ = min(u,v)
                max_ = max(u,v)
                u = min_
                v = max_
                arr = np.concatenate([F[u,:],F[v,:]])
                E[i,:] = np.random.multivariate_normal(arr @ self.weight_centers,
                                                       np.eye(D))
                G[u][v]["weight"] = E[i,:] 
                i+=1
    
        return G,Y,E,M