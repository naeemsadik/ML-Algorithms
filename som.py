import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


class SelfOrganizingMap:
    def __init__(self, rows=7, columns=7, learning_rate=0.5, sigma=None, iterations=3000, random_state=42):
        self.rows = rows
        self.columns = columns
        self.learning_rate = learning_rate
        self.sigma = sigma or max(rows, columns) / 2
        self.iterations = iterations
        self.random_state = random_state

    def _winner(self, sample):
        distances = np.linalg.norm(self.weights_ - sample, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        rng = np.random.default_rng(self.random_state)
        choices = rng.choice(len(x), self.rows * self.columns, replace=True)
        self.weights_ = x[choices].reshape(self.rows, self.columns, x.shape[1]).copy()
        grid = np.indices((self.rows, self.columns)).transpose(1, 2, 0)
        for step in range(self.iterations):
            sample = x[rng.integers(len(x))]
            winner = np.array(self._winner(sample))
            progress = step / self.iterations
            rate = self.learning_rate * np.exp(-progress)
            radius = max(self.sigma * np.exp(-progress), 1e-6)
            grid_distance = np.sum((grid - winner) ** 2, axis=2)
            influence = np.exp(-grid_distance / (2 * radius ** 2))[..., None]
            self.weights_ += rate * influence * (sample - self.weights_)
        return self

    def transform(self, x):
        return np.array([self._winner(sample) for sample in np.asarray(x, dtype=float)])

    def quantization_error(self, x):
        x = np.asarray(x, dtype=float)
        winners = self.transform(x)
        mapped = self.weights_[winners[:, 0], winners[:, 1]]
        return float(np.mean(np.linalg.norm(x - mapped, axis=1)))


def demo():
    x, labels = make_blobs(n_samples=360, centers=4, cluster_std=0.65, random_state=51)
    x = StandardScaler().fit_transform(x)
    model = SelfOrganizingMap(rows=7, columns=7, iterations=3500, random_state=51).fit(x)
    positions = model.transform(x)
    cluster_centers = np.array([positions[labels == label].mean(axis=0) for label in np.unique(labels)])
    error = model.quantization_error(x)
    assert error < 0.25
    assert len(np.unique(np.rint(cluster_centers), axis=0)) == 4
    print(cluster_centers)
    print(f"quantization_error={error:.3f}")


if __name__ == "__main__":
    demo()
