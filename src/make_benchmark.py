"""
USO:
python make_benchmark.py "path_to_save_benchmark_files"
"""

import os
import sys
from attrAgm import *
import networkx as nx
import shutil

path="./"
if len(sys.argv)>1:
    path = sys.argv[1]

try:
    os.remove("attr.npy")
    os.remove("E.npy")
    os.remove("agm.txt")
    os.remove("membership_matrix.npy")
except:
    pass

K=3
D=4 
attrAGM().gen_bipartite_net(num_nodes=30000,n_clusters=K,bipartite_prob=0.2)
### Single attribute at edges
lambdas = [str(round(i,2)) for i in np.linspace(1,100,K)] ##make K+1 with pn > 0 np.random.rand(K+1)
print("\nLAMBDAS:",lambdas)
G,Y,E,M = attrAGM(radius=100).generate_benchmark(n_steps=20,lambdas=lambdas,probs=[0.3]*K,pn=-1)
### Multidimension attributes at edges
# WC = np.random.rand(2*K,D)
# benchmark = attrAGM(weight_centers=WC,radius=100)
# G,Y,E,M = benchmark.generate_benchmark(n_steps=20,probs=[0.3]*K,pn=1)
# print("\nThe centers for attributes are: \n",benchmark.att_centers)
####
with open(path+'attr.npy', 'wb') as f:
    np.save(f, Y)
with open(path+'membership_matrix.npy', 'wb') as f:
    np.save(f, M)
if E is not None:
    with open(path+'E.npy', 'wb') as f:
        np.save(f, E)
nx.write_weighted_edgelist(G,path+'net.txt',delimiter=',')
shutil.move("agm.txt",path+"agm.txt")
shutil.move("bipartite.txt",path+"bipartite.txt")