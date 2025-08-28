"""
We extend the model to include attributes as well
Assuming that the covariance matrix is the identity, the loglikelihood for 
the attributes is 

        h(F_u) = -1/2 * norm2(Y_u' - F_u'*A)^2 

and the derivative is:
        
        d h(F_u)/df = A*(x-A'*f)
        
For the edges counter part: 
        
        d/df -1/2 * norm2(x' - f'*A - v')^2 = A*(x-A'*f-v)
"""

import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt

class attrAgmFit():
    """
    MinVal/MaxVal are the bounds for the F matrix. The upper bound is solely for numeric reasons 
    """
    def __init__(self, 
                 MinVal=0,
                 MaxVal=1000,
                 num_communities = 3,
                 update_F_iterations = 5,
                 update_F_lr=0.005,
                 display_loss=False, 
                 use_line_search=True,
                 line_search_Alpha=0.05,
                 line_search_Beta=0.3,
                 line_search_MaxIter=5):
        self.MinVal = MinVal
        self.MaxVal = MaxVal
        self.n_clusters = num_communities
        self.N = None #number of data points
        self.D_attr = None #the dimension of attributes
        self.D_weight = None #the dimension of weights

        self.update_F_iterations = update_F_iterations
        self.update_F_lr = update_F_lr
        self.update_F_use_line_search = use_line_search
        self.display_ll = display_loss
        
        self.line_search_Alpha = line_search_Alpha
        self.line_search_Beta = line_search_Beta
        self.line_search_MaxIter = line_search_MaxIter
        self.sum_Fv = None
        
    def sigm(sel, x):
        term = None
        try:
            term = np.divide(np.exp(-1.*x),1.-np.exp(-1.*x))
        except:
            term = np.zeros(x.shape)
        return term

    def log_likelihood_row(self, F, G, Y, A, B, u):
        total_edges = 0
        for v in G.neighbors(u):
            total_edges += np.log(1 - np.exp( -F[u].dot(F[v]) ) )
        
        total_non_edges = 0
        # for v in nx.non_neighbors(G,u):
        #     total_non_edges += F[u].dot(F[v])
        sum_Fv = self.sum_Fv.copy()
        for v in G.neighbors(u):
             sum_Fv -= F[v]
        sum_Fv -= F[u]
        total_non_edges = F[u].dot(sum_Fv)
        graph_ll = total_edges - total_non_edges
        attr_ll = - 1/2 * np.linalg.norm(Y[u,:] - F[u,:]@A) ** 2
        
        ## Add the weights part
        weights_ll = 0
        for u,v in G.edges:
            min_ = min(u,v)
            max_ = max(u,v)
            u = min_
            v = max_
            arr = np.concatenate([F[u,:],F[v,:]])
            weights_ll += - 1/2 * np.linalg.norm( G[u][v]["weight"] - arr@B) ** 2   
        total_ll = 0
        total_ll += graph_ll
        total_ll += attr_ll
        total_ll += weights_ll
        return total_ll 

    def log_likelihood(self, F, G, Y, A, B):
        """implements equation 2 of 
        https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf"""
        total = np.sum([self.log_likelihood_row(F, G, Y, A, B, u) for u in G.nodes()])
        return total

    def line_search(self, F, G, Y, A, B, u, DeltaV,GradV):
        StepSize = 1.0
        InitLikelihood = self.log_likelihood_row(F, G, Y, A, B, u)
        NewVarV = np.zeros_like(DeltaV)
        for i in range(self.line_search_MaxIter):
            for j in range(NewVarV.shape[0]):
                NewVal = F[u,j] + StepSize * DeltaV[j]
                if (NewVal < self.MinVal):
                    NewVal = self.MinVal
                if (NewVal > self.MaxVal):
                    NewVal = self.MaxVal
                NewVarV[j]= NewVal
            F_new=F.copy()   
            F_new[u,:] = NewVarV 
            if self.log_likelihood_row(F_new, G, Y, A, B, u) < InitLikelihood + self.line_search_Alpha * StepSize * GradV.dot(DeltaV):
                StepSize *= self.line_search_Beta
            else:
                break
            if (i == self.line_search_MaxIter - 1):
                StepSize = 0
                break
        return StepSize

    def update_A(self, F, Y):
        # grad_A = np.zeros((self.n_clusters, self.D_attr))
        """
        A = (M^T @ M)^-1 @ M^T @ Y
        np.linalg.solve(F,Y)
        """
        return np.linalg.inv(F.T @ F) @ F.T @ Y

    def update_B(self, F, G):
        stacked_arr = np.zeros((G.number_of_edges(),2*self.n_clusters))
        stacked_edges = np.zeros((G.number_of_edges(),self.D_weight))
        i=0
        for u,v in G.edges:
            min_ = min(u,v)
            max_ = max(u,v)
            u = min_
            v = max_
            arr = np.concatenate([F[u,:],F[v,:]])  
            stacked_arr[i,:] = arr
            stacked_edges[i,:] = G[u][v]["weight"]
            i+=1
        return np.linalg.inv(stacked_arr.T @ stacked_arr) @ stacked_arr.T @ stacked_edges
        #return np.linalg.solve(stacked_arr,stacked_edges)
        
        """
        grad_B = np.zeros(B.shape)
        for u,v in G.edges:
            min_ = min(u,v)
            max_ = max(u,v)
            u = min_
            v = max_
            arr = np.concatenate([F[u,:],F[v,:]])  
            term = G[u][v]["weight"] - arr@B
            grad_B += np.outer(arr, term)
        return B + lr*grad_B
        """

    """
    Input:

    F is the current estimated membership matrix

    G is a networkx graph

    Y is a N x D array of attributes

    A is k x D_attributes array of attributes centers

    B is 2k x D_weights of weights centers

    u is the current vertex index

    self.sum_Fv is k dimensional vector, containing the sum of: 
        np.sum([F[v,:] for v in G.nodes()],axis=1) 
    """
    def gradient_efficient(self, F, G, Y, A, B, u):
        """Implements equation 3 of
        https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
        """
        
        sum_neigh = np.zeros((self.n_clusters,))
        for v in G.neighbors(u):
            dotproduct = F[v].dot(F[u])
            sum_neigh += F[v]*self.sigm(dotproduct)

        sum_nneigh = self.sum_Fv.copy()
        #Speed up this computation using eq.4
        for v in nx.neighbors(G,u):
            sum_nneigh -= F[v]
        sum_nneigh -= F[u]
        grad = np.zeros(self.n_clusters)
        grad_net = sum_neigh - sum_nneigh
        ## Add the attributes part
        grad_att = A@(Y[u,:]- (A.T@F[u,:]))
        ## Add the weights part
        ## we decompose the [Fu,F_v]@B into F[u] @ B + constant_vector
        total = constant_vec = np.zeros(self.D_weight )
        grad_weights = np.zeros(self.n_clusters)
        for v in G.neighbors(u):
            if v > u:
                total = B[:self.n_clusters,:].T @ F[u,:]
                constant_vec = B[self.n_clusters:,:].T @ F[v,:]
                #print(B[:C,:].shape,total.shape, constant_vec.shape,  G[u][v]["weight"].shape)
                grad_weights += B[:self.n_clusters,:] @ ( G[u][v]["weight"] - total - constant_vec)
            else:
                ## Just invert the indexing of B
                total = B[self.n_clusters:,:].T@F[u,:]
                constant_vec = B[:self.n_clusters,:].T@F[v,:]
                grad_weights += ( B[self.n_clusters:,:] @ ( G[u][v]["weight"] - total - constant_vec) )    
        grad += grad_net
        grad += grad_att
        grad += grad_weights                
        grad = np.clip(grad,-10,10)
        return grad

    """
    Previously train_efficient
    """
    def update_F(self, G, Y, A, B, F_init=None):
        F = None
        if F_init is not None:
            F = F_init
        else:
            # initialize an F
            F = np.random.rand(G.number_of_nodes(),self.n_clusters)
        
        # self.sum_Fv = F.sum(axis=0)    
        # self.sum_Fv = np.zeros((self.n_clusters,))
        # for u in G.nodes():
        #     self.sum_Fv += F[u,:]
            
        for n in range(self.update_F_iterations):
            for u in G.nodes():
                grad = self.gradient_efficient(F, G, Y, A, B, u)
                prev_Fu = F[u,:]
                if self.update_F_use_line_search:
                    alpha = self.line_search(F, G, Y, A, B, u, grad, grad)
                    F[u] += alpha*grad
                else:
                    F[u] += self.update_F_lr*grad
                F[u,:] = np.maximum(0.001, F[u,:]) # F should be nonnegative
                new_Fu = F[u,:]
                delta_Fu = (new_Fu - prev_Fu)
                self.sum_Fv += delta_Fu
        return F

    def fit_MLE(self, G, Y, C=3, iterations = 100, F_init=None, Thres=0.001):
        self.n_clusters = C
        self.N, self.D_attr = Y.shape
        self.D_weight = list(G.edges(data=True))[0][2]["weight"].shape[0]
        F = None
        if F_init is not None:
            F = F_init
        else:
            # initialize an F
            F = np.random.rand(G.number_of_nodes(),self.n_clusters)
        self.sum_Fv = np.sum(F,axis=0)
        A = self.update_A(F, Y)
        B = self.update_B(F, G)
        losses = []
        Prev_ll = self.log_likelihood(F, G, Y, A, B)
        Cur_ll = Prev_ll + Thres*np.fabs(Prev_ll) + 2
        i = 0
        while (i < iterations) and ( Cur_ll - Prev_ll > Thres*np.fabs(Prev_ll) ):
            F = self.update_F(G, Y, A, B, F_init=F)
            A = self.update_A(F, Y)
            B = self.update_B(F, G)
            Cur_ll = self.log_likelihood(F, G, Y, A, B)
            if self.display_ll:
                losses.append(Cur_ll)
            i+=1
        if self.display_ll:
            xs = [x for x in range(len(losses))]
            plt.plot(xs, losses)
            plt.show()
            # Make sure to close the plt object once done
            plt.close()
        return F,A,B    

"""
###############################################################################################################
                
                DEPRECATED METHODS
                
###############################################################################################################       
"""

# def gradient(F, G, Y, A, B, u):
#     """Implements equation 3 of
#     https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
    
#       * i indicates the row under consideration
    
#     The many forloops in this function can be optimized, but for
#     educational purposes we write them out clearly
#     """
#     N, C = F.shape

#     sum_neigh = np.zeros((C,))
#     for v in G.neighbors(u):
#         dotproduct = F[v].dot(F[u])
#         sum_neigh += F[v]*sigm(dotproduct)

#     sum_nneigh = np.zeros((C,))
#     #Speed up this computation using eq.4
#     for v in nx.non_neighbors(G,u):
#         sum_nneigh += F[v]

#     grad = sum_neigh - sum_nneigh
#     grad_net = sum_neigh - sum_nneigh
#     ## Add the attributes part
#     grad_att = A@(Y[u,:]- (A.T@F[u,:]))
#     ## Add the weights part
#     ## we decompose the [Fu,F_v]@B into F[u] @ B + constant_vector
#     total = constant_vec = np.zeros(D)
#     grad_weights = np.zeros(C)
#     for v in G.neighbors(u):
#         if v > u:
#             total = B[:C,:].T @ F[u,:]
#             constant_vec = B[C:,:].T @ F[v,:]
#             #print(B[:C,:].shape,total.shape, constant_vec.shape,  G[u][v]["weight"].shape)
#             grad_weights += B[:C,:] @ ( G[u][v]["weight"] - total - constant_vec)
#         else:
#             ## Just invert the indexing of B
#             total = B[C:,:].T@F[u,:]
#             constant_vec = B[:C,:].T@F[v,:]
#             grad_weights += ( B[C:,:] @ ( G[u][v]["weight"] - total - constant_vec) )    
#     grad += grad_net
#     grad += grad_att
#     grad += grad_weights
#     grad = np.clip(grad,-10,10)
#     return grad

# def train(G, Y, A, B, C, iterations = 100,lr=0.005, display_loss=False, F_init=None, use_line_search=False):
#     F = None
#     if F_init is not None:
#         F = F_init
#     else:
#         # initialize an F
#         F = np.random.rand(G.number_of_nodes(),C)
#     losses = []
#     for n in range(iterations):
#         for u in G.nodes():
#             grad = gradient(F, G, Y, A, B, u)
#             if use_line_search:
#                 alpha = line_search(F, G, Y, A, B, u, grad,grad,0.05,0.3,5)
#                 F[u] += alpha*grad
#             else:
#                 F[u] += lr*grad
#             F[u] = np.maximum(0.001, F[u]) # F should be nonnegative
#         if display_loss:
#             ll = log_likelihood(F, G, Y, A, B)
#             losses.append(ll)
#             #print('At step %5i/%5i ll is %5.3f'%(n, iterations, ll))
#     if display_loss:
#         xs = [x for x in range(len(losses))]
#         plt.plot(xs, losses)
#         plt.show()
#         # Make sure to close the plt object once done
#         plt.close()
#     return F