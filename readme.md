# EgoSplit-sknetwork

This package provides a fast and flexible implementation of the egosplitting community detection paradigm for detecting overlapping communities.
For details and motivation of the algorithm, please see the paper below.
The reference implementation is available [here](https://github.com/google-research/google-research/blob/master/graph_embedding/persona/persona.py).


> Alessandro Epasto, Silvio Lattanzi, and Renato Paes Leme. 2017. Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17). Association for Computing Machinery, New York, NY, USA, 145-154. https://doi.org/10.1145/3097983.3098054


# Example

```python
import sknetwork as sn
from egosplit_sknetwork import EgoSplit

g = sn.data.toy_graphs.karate_club()
egosplit = EgoSplit()
labels = egosplit.fit_predict(g)
```

To pass other clustering algorithms to egosplit, they must be initialized in advace and passed as parameters.
The algorithm accepts any subclass of sknetwork.clustering.BaseClustering for either local_clustering (used to cluster the egonets) or the global_clustering (use to cluster the persona graph).

```python
high_res_clusterer = sn.clustering.Louvain(resolution=5, random_state=42)
egosplit = EgoSplit(local_clustering='PC', global_clustering=high_res_clusterer)
labels = egosplit.fit_predict(g)
```