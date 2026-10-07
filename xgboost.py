import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class XGBoostRegressor:
    def __init__(self, n_estimators=200, learning_rate=0.08, regularization=1.0, gamma=0.0, max_bins=32):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.gamma = gamma
        self.max_bins = max_bins

    def _best_stump(self, x, gradients):
        total_gradient = gradients.sum()
        total_count = len(x)
        best = None
        best_gain = -np.inf
        for feature in range(x.shape[1]):
            values = x[:, feature]
            thresholds = np.unique(np.quantile(values, np.linspace(0.05, 0.95, self.max_bins)))
            for threshold in thresholds:
                left = values <= threshold
                left_count = int(left.sum())
                right_count = total_count - left_count
                if left_count == 0 or right_count == 0:
                    continue
                left_gradient = gradients[left].sum()
                right_gradient = total_gradient - left_gradient
                gain = 0.5 * (
                    left_gradient ** 2 / (left_count + self.regularization)
                    + right_gradient ** 2 / (right_count + self.regularization)
                    - total_gradient ** 2 / (total_count + self.regularization)
                ) - self.gamma
                if gain > best_gain:
                    left_weight = -left_gradient / (left_count + self.regularization)
                    right_weight = -right_gradient / (right_count + self.regularization)
                    best = feature, float(threshold), float(left_weight), float(right_weight)
                    best_gain = gain
        return best

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.base_score_ = float(y.mean())
        predictions = np.full(len(y), self.base_score_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            gradients = predictions - y
            stump = self._best_stump(x, gradients)
            if stump is None:
                break
            self.trees_.append(stump)
            feature, threshold, left_weight, right_weight = stump
            predictions += self.learning_rate * np.where(x[:, feature] <= threshold, left_weight, right_weight)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        predictions = np.full(len(x), self.base_score_)
        for feature, threshold, left_weight, right_weight in self.trees_:
            predictions += self.learning_rate * np.where(x[:, feature] <= threshold, left_weight, right_weight)
        return predictions


def demo():
    x, y = make_regression(n_samples=650, n_features=8, n_informative=7, noise=10, random_state=36)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=36)
    model = XGBoostRegressor(n_estimators=220, learning_rate=0.08).fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = r2_score(y_test, predictions)
    assert score > 0.8
    print(f"trees={len(model.trees_)}")
    print(f"rmse={mean_squared_error(y_test, predictions) ** 0.5:.3f}")
    print(f"r2={score:.3f}")


if __name__ == "__main__":
    demo()
