import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


class HierarchicalClustering:
    def __init__(self, n_clusters=3, method="ward"):
        self.n_clusters = n_clusters
        self.method = method

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.linkage_matrix_ = linkage(x, method=self.method)
        self.labels_ = fcluster(self.linkage_matrix_, self.n_clusters, criterion="maxclust") - 1
        return self

    def fit_predict(self, x):
        return self.fit(x).labels_


def demo():
    x, expected = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=14)
    model = HierarchicalClustering(n_clusters=3).fit(x)
    score = adjusted_rand_score(expected, model.labels_)
    assert score > 0.95
    print(model.linkage_matrix_[-3:])
    print(f"adjusted_rand_score={score:.3f}")


if __name__ == "__main__":
    demo()
