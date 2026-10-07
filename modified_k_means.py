import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


class ModifiedKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tolerance=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

    def _initialize(self, x):
        rng = np.random.default_rng(self.random_state)
        centroids = [x[rng.integers(len(x))]]
        while len(centroids) < self.n_clusters:
            distances = np.min(np.sum((x[:, None, :] - np.array(centroids)[None, :, :]) ** 2, axis=2), axis=1)
            probabilities = distances / distances.sum()
            centroids.append(x[rng.choice(len(x), p=probabilities)])
        return np.array(centroids)

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.centroids_ = self._initialize(x)
        for _ in range(self.max_iter):
            labels = self.predict(x)
            updated = self.centroids_.copy()
            errors = np.sum((x - self.centroids_[labels]) ** 2, axis=1)
            for cluster in range(self.n_clusters):
                members = x[labels == cluster]
                updated[cluster] = members.mean(axis=0) if len(members) else x[np.argmax(errors)]
            if np.linalg.norm(updated - self.centroids_) <= self.tolerance:
                self.centroids_ = updated
                break
            self.centroids_ = updated
        self.labels_ = self.predict(x)
        self.inertia_ = float(np.sum((x - self.centroids_[self.labels_]) ** 2))
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        distances = np.sum((x[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1)


def demo():
    x, expected = make_blobs(n_samples=240, centers=4, cluster_std=0.7, random_state=11)
    model = ModifiedKMeans(n_clusters=4, random_state=11).fit(x)
    score = adjusted_rand_score(expected, model.labels_)
    assert score > 0.95
    print(model.centroids_)
    print(f"adjusted_rand_score={score:.3f}")


if __name__ == "__main__":
    demo()
