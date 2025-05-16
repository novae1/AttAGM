Attributed AGM code

## File description
### attrAGM.py
In this file we specify how to generate the benchmark. The graph is constructed in the same way as in
<a href="https://github.com/FelipeSchreiber/snap"> Affiliation Graph Model </a>. The edge weights are generated either via mixture of exponentials or from ~N([Fu,Fv]*W_1). The attributes are sampled from ~N(F*W_2, I).

### make_benchmark.py
Simply call the attrAGM.py script with some default arguments and save the generated instance in matrix format

### bigclam.py
A simple implementation of the <a href="http://i.stanford.edu/~crucis/pubs/paper-nmfagm.pdf">BigClam algorithm</a>

### banerjee_overlapping.py
In this file we implement the algorithm <a href="file:///home/felipe/Downloads/banerjee05overlapping.pdf">Model-based Overlapping Clustering</a> along with some variations.

### Pc2Fu.ipynb
A file to test if the scripts attrAGM.py and bigclam.py are working well

### banerjee_overlapping.jl
In this file we implement the algorithm <a href="file:///home/felipe/Downloads/banerjee05overlapping.pdf">Model-based Overlapping Clustering</a> along with some variations and using JuMP solvers.

### banerjee_julia.ipynb
A file to test if the script banerjee_overlapping.jl is working well

### BigClam.jl
A simple implementation of the <a href="http://i.stanford.edu/~crucis/pubs/paper-nmfagm.pdf">BigClam algorithm</a> using JuMP solvers.
