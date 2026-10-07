import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def demo():
    x, expected = make_blobs(
        n_samples=[80, 120, 160],
        centers=[[-5, -3], [0, 4], [5, -2]],
        cluster_std=[0.3, 0.7, 1.1],
        random_state=24,
    )
    rng = np.random.default_rng(24)
    noise = rng.uniform(-10, 10, size=(25, 2))
    x = np.vstack([x, noise])
    expected = np.concatenate([expected, np.full(len(noise), -1)])
    model = HDBSCAN(min_cluster_size=25, min_samples=8, cluster_selection_method="eom").fit(x)
    score = adjusted_rand_score(expected, model.labels_)
    assert score > 0.85
    print(np.unique(model.labels_, return_counts=True))
    print(f"adjusted_rand_score={score:.3f}")


if __name__ == "__main__":
    demo()
