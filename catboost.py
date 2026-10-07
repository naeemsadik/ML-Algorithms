import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class CatBoostRegressor:
    def __init__(self, cat_features, n_estimators=180, learning_rate=0.06, max_depth=3, smoothing=8.0, random_state=42):
        self.cat_features = set(cat_features)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.smoothing = smoothing
        self.random_state = random_state

    def _fit_transform(self, x, y):
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(x))
        transformed = np.empty(x.shape, dtype=float)
        self.category_values_ = {}
        for feature in range(x.shape[1]):
            if feature not in self.cat_features:
                transformed[:, feature] = x[:, feature].astype(float)
                continue
            sums = {}
            counts = {}
            for row in order:
                value = x[row, feature]
                transformed[row, feature] = (sums.get(value, 0.0) + self.smoothing * self.prior_) / (counts.get(value, 0) + self.smoothing)
                sums[value] = sums.get(value, 0.0) + y[row]
                counts[value] = counts.get(value, 0) + 1
            self.category_values_[feature] = {
                value: (sums[value] + self.smoothing * self.prior_) / (counts[value] + self.smoothing)
                for value in sums
            }
        return transformed

    def _transform(self, x):
        transformed = np.empty(x.shape, dtype=float)
        for feature in range(x.shape[1]):
            if feature not in self.cat_features:
                transformed[:, feature] = x[:, feature].astype(float)
                continue
            mapping = self.category_values_[feature]
            transformed[:, feature] = [mapping.get(value, self.prior_) for value in x[:, feature]]
        return transformed

    def fit(self, x, y):
        x = np.asarray(x, dtype=object)
        y = np.asarray(y, dtype=float)
        self.prior_ = float(y.mean())
        encoded = self._fit_transform(x, y)
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            loss="huber",
            random_state=self.random_state,
        ).fit(encoded, y)
        return self

    def predict(self, x):
        return self.model_.predict(self._transform(np.asarray(x, dtype=object)))


def demo():
    rng = np.random.default_rng(42)
    cities = rng.choice(["Dhaka", "Chattogram", "Khulna", "Rajshahi"], size=800)
    plans = rng.choice(["basic", "standard", "premium"], size=800)
    usage = rng.uniform(1, 30, size=800)
    city_effect = {"Dhaka": 18, "Chattogram": 12, "Khulna": 7, "Rajshahi": 9}
    plan_effect = {"basic": 15, "standard": 35, "premium": 70}
    y = np.array([city_effect[city] + plan_effect[plan] for city, plan in zip(cities, plans)]) + 3.5 * usage + rng.normal(0, 3, size=800)
    x = np.column_stack([cities, plans, usage])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    model = CatBoostRegressor(cat_features=[0, 1], random_state=42).fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = r2_score(y_test, predictions)
    assert score > 0.9
    print(f"rmse={mean_squared_error(y_test, predictions) ** 0.5:.3f}")
    print(f"r2={score:.3f}")


if __name__ == "__main__":
    demo()
