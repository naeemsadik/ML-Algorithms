import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


class FuzzyCMeans:
    def __init__(self, n_clusters=3, fuzziness=2.0, max_iter=300, tolerance=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

    def _memberships(self, x):
        distances = np.linalg.norm(x[:, None, :] - self.centroids_[None, :, :], axis=2)
        distances = np.maximum(distances, np.finfo(float).eps)
        weights = distances ** (-2 / (self.fuzziness - 1))
        return weights / weights.sum(axis=1, keepdims=True)

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(self.random_state)
        memberships = rng.random((len(x), self.n_clusters))
        memberships /= memberships.sum(axis=1, keepdims=True)
        for _ in range(self.max_iter):
            powered = memberships ** self.fuzziness
            self.centroids_ = powered.T @ x / powered.sum(axis=0)[:, None]
            updated = self._memberships(x)
            if np.max(np.abs(updated - memberships)) <= self.tolerance:
                memberships = updated
                break
            memberships = updated
        self.membership_ = memberships
        self.labels_ = np.argmax(memberships, axis=1)
        return self

    def predict(self, x):
        return np.argmax(self._memberships(np.asarray(x, dtype=float)), axis=1)


def demo():
    x, expected = make_blobs(n_samples=180, centers=3, cluster_std=0.65, random_state=18)
    model = FuzzyCMeans(n_clusters=3, random_state=18).fit(x)
    score = adjusted_rand_score(expected, model.labels_)
    assert score > 0.95
    assert np.allclose(model.membership_.sum(axis=1), 1)
    print(model.centroids_)
    print(f"adjusted_rand_score={score:.3f}")


if __name__ == "__main__":
    demo()
