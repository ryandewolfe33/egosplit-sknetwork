# EgoSplit-sknetwork

This package provides a fast and flexible implementation of the egosplitting community detection paradigm for detecting overlapping communities.
For details and motivation of the algorithm, please see the paper below.
The reference implementation is available [here](https://github.com/google-research/google-research/blob/master/graph_embedding/persona/persona.py).


> Alessandro Epasto, Silvio Lattanzi, and Renato Paes Leme. 2017. Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17). Association for Computing Machinery, New York, NY, USA, 145-154. https://doi.org/10.1145/3097983.3098054

# Installation

Currently you can install this package by cloning this repository and installing locally.
```sh
git clone https://github.com/ryandewolfe33/egosplit-sknetwork.git
cd egosplit-sknetwork
pip install .
```


# Example

```python
import sknetwork as sn
from egosplit_sknetwork import EgoSplit

g = sn.data.toy_graphs.karate_club()
egosplit = EgoSplit()
labels = egosplit.fit_predict(g)
```

By default the algorithm uses [Propagation Clustering](https://scikit-network.readthedocs.io/en/latest/reference/clustering.html#sknetwork.clustering.PropagationClustering) for local clustering and [Leiden](https://scikit-network.readthedocs.io/en/latest/reference/clustering.html#sknetwork.clustering.Leiden) for global clustering.
To pass other clustering algorithms to egosplit, they must be initialized in advace and passed as parameters.
The algorithm accepts any subclass of [sknetwork.clustering.BaseClustering](https://scikit-network.readthedocs.io/en/latest/reference/clustering.html) for either local_clustering (used to cluster the egonets) or global_clustering (used to cluster the persona graph).

```python
high_res_clusterer = sn.clustering.Louvain(resolution=5, random_state=42)
egosplit = EgoSplit(local_clustering='PC', global_clustering=high_res_clusterer)
labels = egosplit.fit_predict(g)
```

Labels is a sparse matrix with dimensions (n_labels, n_vertices), where `labels[i,j] = True` if vertex j is in cluster i.