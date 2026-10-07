import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class GeneralizedRegressionNeuralNetwork:
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def fit(self, x, y):
        self.x_train_ = np.asarray(x, dtype=float)
        self.y_train_ = np.asarray(y, dtype=float)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        squared_distances = np.sum((x[:, None, :] - self.x_train_[None, :, :]) ** 2, axis=2)
        weights = np.exp(-squared_distances / (2 * self.sigma ** 2))
        return weights @ self.y_train_ / np.maximum(weights.sum(axis=1), np.finfo(float).eps)


def demo():
    rng = np.random.default_rng(63)
    x = rng.uniform(-3, 3, size=(600, 2))
    y = np.sin(1.7 * x[:, 0]) + 0.4 * x[:, 1] ** 2 + rng.normal(0, 0.08, size=len(x))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=63)
    scaler = StandardScaler().fit(x_train)
    model = GeneralizedRegressionNeuralNetwork(sigma=0.25).fit(scaler.transform(x_train), y_train)
    predictions = model.predict(scaler.transform(x_test))
    score = r2_score(y_test, predictions)
    assert score > 0.9
    print(f"rmse={mean_squared_error(y_test, predictions) ** 0.5:.3f}")
    print(f"r2={score:.3f}")


if __name__ == "__main__":
    demo()
