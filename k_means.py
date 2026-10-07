import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tolerance=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(x), self.n_clusters, replace=False)
        self.centroids_ = x[indices].copy()
        for _ in range(self.max_iter):
            labels = self.predict(x)
            updated = np.array([
                x[labels == cluster].mean(axis=0) if np.any(labels == cluster) else self.centroids_[cluster]
                for cluster in range(self.n_clusters)
            ])
            if np.linalg.norm(updated - self.centroids_) <= self.tolerance:
                self.centroids_ = updated
                break
            self.centroids_ = updated
        self.labels_ = self.predict(x)
        self.inertia_ = float(np.sum((x - self.centroids_[self.labels_]) ** 2))
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        distances = np.linalg.norm(x[:, None, :] - self.centroids_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)


def demo():
    x, expected = make_blobs(n_samples=180, centers=3, cluster_std=0.55, random_state=7)
    model = KMeans(n_clusters=3, random_state=7).fit(x)
    score = adjusted_rand_score(expected, model.labels_)
    assert score > 0.95
    print(model.centroids_)
    print(f"adjusted_rand_score={score:.3f}")


if __name__ == "__main__":
    demo()
