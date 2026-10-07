import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


class DBSCAN:
    def __init__(self, eps=0.3, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        neighborhoods = [np.flatnonzero(row <= self.eps) for row in distances]
        labels = np.full(len(x), -1, dtype=int)
        visited = np.zeros(len(x), dtype=bool)
        cluster = 0
        for point in range(len(x)):
            if visited[point]:
                continue
            visited[point] = True
            neighbors = neighborhoods[point]
            if len(neighbors) < self.min_samples:
                continue
            labels[point] = cluster
            seeds = list(neighbors[neighbors != point])
            queued = set(seeds)
            while seeds:
                neighbor = seeds.pop()
                if not visited[neighbor]:
                    visited[neighbor] = True
                    nearby = neighborhoods[neighbor]
                    if len(nearby) >= self.min_samples:
                        for candidate in nearby:
                            if candidate not in queued:
                                seeds.append(int(candidate))
                                queued.add(int(candidate))
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster
            cluster += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.array([i for i, neighbors in enumerate(neighborhoods) if len(neighbors) >= self.min_samples])
        return self

    def fit_predict(self, x):
        return self.fit(x).labels_


def demo():
    x, expected = make_moons(n_samples=240, noise=0.05, random_state=21)
    x = StandardScaler().fit_transform(x)
    model = DBSCAN(eps=0.28, min_samples=5).fit(x)
    score = adjusted_rand_score(expected, model.labels_)
    assert score > 0.95
    print(np.unique(model.labels_, return_counts=True))
    print(f"adjusted_rand_score={score:.3f}")


if __name__ == "__main__":
    demo()
